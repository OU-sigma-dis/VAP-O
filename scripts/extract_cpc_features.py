"""CPC特徴量（gEncoder + gAR出力、100Hz）を事前抽出してディスクに保存するスクリプト。

入力: CSVファイル（session, audio_path列を含む）
出力: {session}_cpc.pt ファイル（feat_1, feat_2 を含む辞書、100Hz・256次元）
役割: 事前学習済みCPCモデルの確定的な出力のみを保存する。
      ダウンサンプラー（100Hz→20Hz）はランダム初期化のため含めず、学習時にモデル内で適用する。

メモリ改善:
  - セッションごとにメモリ解放（gc.collect）
  - 長い音声はチャンク分割して処理（GRUの中間状態によるメモリ膨張を防止）
  - 各チャネル（話者）を個別に処理してピークメモリを半減
  - 処理済みセッションを自動スキップ（再開対応）
  - MPS/CUDA対応で高速化

使い方:
  python scripts/extract_cpc_features.py \
    --csv_paths /path/to/vap-o_dataset/train.csv \
                /path/to/vap-o_dataset/val.csv \
                /path/to/vap-o_dataset/test.csv \
    --output_dir /path/to/vap-o_dataset/cpc_features \
    --chunk_sec 60
"""

import argparse
import gc
import sys
from pathlib import Path

import einops
import pandas as pd
import torch
import torch.nn as nn

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from models.encoder_components import load_CPC
from utilities.audio import load_waveform


# CPCのダウンサンプリング比率（16kHz音声 → 100Hz特徴量）
CPC_DOWNSAMPLE_FACTOR = 160
SAMPLE_RATE = 16_000


def get_device() -> torch.device:
    """利用可能な最適デバイスを返す。

    出力: torch.device（MPS > CUDA > CPU の優先順）
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_cpc_single_channel(
    waveform_mono: torch.Tensor,
    cpc_model: nn.Module,
    device: torch.device,
    chunk_sec: float = 60.0,
) -> torch.Tensor:
    """1チャネルの音声からCPC特徴量（100Hz）を抽出する。

    入力:
      waveform_mono: (1, n_samples) のモノラル音声（CPU上）
      cpc_model: CPCModel（gEncoder + gAR）、device上に配置済み
      device: 計算デバイス
      chunk_sec: チャンク分割の秒数（メモリ制限用）
    出力: (n_frames_100hz, 256) の特徴量テンソル（CPU上）
    """
    n_samples = waveform_mono.shape[1]
    total_sec = n_samples / SAMPLE_RATE

    if total_sec <= chunk_sec:
        # 短い音声はそのまま処理
        x = waveform_mono.unsqueeze(0).to(device)  # (1, 1, n_samples)
        z = cpc_model.gEncoder(x)
        z = einops.rearrange(z, "b c n -> b n c")
        z = cpc_model.gAR(z)
        feat = z.squeeze(0).cpu()  # (n_frames, 256)
        del z, x
        return feat

    # 長い音声はチャンク分割して処理
    # GRUの隠れ状態を引き継ぎながら処理する
    chunk_samples = int(chunk_sec * SAMPLE_RATE)
    # CPC_DOWNSAMPLE_FACTORの倍数に揃える
    chunk_samples = (chunk_samples // CPC_DOWNSAMPLE_FACTOR) * CPC_DOWNSAMPLE_FACTOR

    feats = []
    gru_hidden = None
    offset = 0

    while offset < n_samples:
        end = min(offset + chunk_samples, n_samples)
        chunk = waveform_mono[:, offset:end]

        # 末端がCPC_DOWNSAMPLE_FACTORの倍数でなければパディング
        remainder = chunk.shape[1] % CPC_DOWNSAMPLE_FACTOR
        if remainder != 0:
            pad_size = CPC_DOWNSAMPLE_FACTOR - remainder
            chunk = torch.nn.functional.pad(chunk, (0, pad_size))

        x = chunk.unsqueeze(0).to(device)  # (1, 1, chunk_samples)

        # gEncoder（CNN）
        z = cpc_model.gEncoder(x)
        z = einops.rearrange(z, "b c n -> b n c")

        # gAR（GRU）に隠れ状態を引き継ぐ
        cpc_model.gAR.hidden = gru_hidden
        z = cpc_model.gAR(z)
        # 次チャンクのために隠れ状態を保存
        gru_hidden = cpc_model.gAR.hidden

        feats.append(z.squeeze(0).cpu())

        del z, x, chunk
        offset = end

    feat = torch.cat(feats, dim=0)
    del feats, gru_hidden
    cpc_model.gAR.hidden = None
    return feat


def extract_and_save(
    audio_path: str,
    session: str,
    output_dir: Path,
    cpc_model: nn.Module,
    device: torch.device,
    chunk_sec: float,
) -> bool:
    """1セッションのCPC特徴量を抽出して保存する。

    入力: audio_path, session名, 出力ディレクトリ, CPCモデル, デバイス, チャンク秒数
    出力: 成功したかどうか
    """
    output_path = output_dir / f"{session}_cpc.pt"

    # 既に処理済みならスキップ
    if output_path.exists():
        return True

    try:
        # ステレオ音声をロード（CPU上）
        waveform, sr = load_waveform(audio_path, sample_rate=SAMPLE_RATE)

        if waveform.shape[0] < 2:
            logger.warning(f"セッション {session}: チャネル数が2未満（{waveform.shape[0]}）、スキップ")
            return False

        # チャネル1（話者1）を処理してからメモリ解放
        ch1 = waveform[0:1, :]  # (1, n_samples)
        feat_1 = extract_cpc_single_channel(ch1, cpc_model, device, chunk_sec)
        del ch1

        # チャネル2（話者2）
        ch2 = waveform[1:2, :]  # (1, n_samples)
        feat_2 = extract_cpc_single_channel(ch2, cpc_model, device, chunk_sec)
        del ch2, waveform

        # フレーム数を揃える（短い方に合わせる）
        min_frames = min(feat_1.shape[0], feat_2.shape[0])
        feat_1 = feat_1[:min_frames]
        feat_2 = feat_2[:min_frames]

        # float16で保存（ディスク容量削減、ロード時に.float()でfloat32に戻る）
        torch.save(
            {"feat_1": feat_1.half(), "feat_2": feat_2.half()},
            output_path,
        )

        del feat_1, feat_2
        return True

    except Exception as e:
        logger.error(f"セッション {session} でエラー: {e}")
        # 壊れたファイルがあれば削除
        if output_path.exists():
            output_path.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(description="CPC特徴量の事前抽出（100Hz、ダウンサンプラーなし）")
    parser.add_argument(
        "--csv_paths",
        nargs="+",
        required=True,
        help="CSVファイルパス（複数指定可）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="CPC特徴量の保存先ディレクトリ",
    )
    parser.add_argument(
        "--chunk_sec",
        type=float,
        default=60.0,
        help="チャンク分割の秒数（デフォルト: 60秒）。小さくするとメモリ使用量が減る",
    )
    parser.add_argument(
        "--cpc_model_pt",
        type=str,
        default="",
        help="CPCモデルのチェックポイントパス（空ならデフォルト）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # デバイス選択
    device = get_device()
    logger.info(f"使用デバイス: {device}")

    # 全CSVからユニークなセッションを収集
    all_sessions = {}
    for csv_path in args.csv_paths:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            session = str(row["session"])
            if session not in all_sessions:
                all_sessions[session] = str(row["audio_path"])
    logger.info(f"全セッション数: {len(all_sessions)}")

    # 処理済みセッションを確認
    done_sessions = set(
        f.stem.replace("_cpc", "")
        for f in output_dir.glob("*_cpc.pt")
    )
    remaining = {s: p for s, p in all_sessions.items() if s not in done_sessions}
    logger.info(f"処理済み: {len(done_sessions)}, 残り: {len(remaining)}")

    if not remaining:
        logger.info("全セッション処理済みです")
        return

    # CPCモデルをロード（gEncoder + gAR のみ、ダウンサンプラーなし）
    logger.info("CPCモデルをロード中...")
    cpc_model = load_CPC(args.cpc_model_pt, load_state_dict=True)
    cpc_model.eval()
    cpc_model.to(device)
    for p in cpc_model.parameters():
        p.requires_grad_(False)

    logger.info(f"チャンク分割: {args.chunk_sec}秒")
    logger.info("抽出開始（100Hz、ダウンサンプラーなし）...")

    success_count = 0
    fail_count = 0

    for i, (session, audio_path) in enumerate(remaining.items()):
        result = extract_and_save(
            audio_path, session, output_dir, cpc_model, device, args.chunk_sec
        )

        if result:
            success_count += 1
        else:
            fail_count += 1

        # 進捗表示
        total_done = len(done_sessions) + success_count + fail_count
        total = len(all_sessions)
        if (i + 1) % 10 == 0 or (i + 1) == len(remaining):
            logger.info(
                f"進捗: {total_done}/{total} "
                f"({total_done/total*100:.1f}%) "
                f"[今回: 成功={success_count}, 失敗={fail_count}]"
            )

        # メモリ解放
        gc.collect()

    logger.info(f"完了: 成功={success_count}, 失敗={fail_count}")


if __name__ == "__main__":
    main()
