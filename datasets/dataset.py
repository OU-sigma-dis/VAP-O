import json
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from utilities.audio import load_waveform


def load_df(path):
    """CSVファイルを読み込み、メモリ効率の良いDataFrameを返す。

    入力: CSVファイルパス (audio_path, label_path, start, end, session, dataset, vad_list 列)
    出力: pandas DataFrame
    役割: データ型を明示指定してメモリ使用量を削減する
    """
    dtype_spec = {
        "start": "float32",
        "end": "float32",
        "session": "category",
        "dataset": "category",
    }

    return pd.read_csv(
        path,
        dtype=dtype_spec,
        converters={"vad_list": json.loads},
    )


class VapDataset(Dataset):
    """VAP用データセット。スライディングウィンドウでラベル付き音声とテキストコンテキストを返す。

    入力: CSVパス、ウィンドウ設定、テキストトークン設定
    出力: waveform, va, s_05, s_10, s_20, next_filler_class, text_tokens, text_token_positions を含む辞書
    役割: セッション単位でラベルをキャッシュし、ウィンドウ単位でデータを切り出す
    """

    def __init__(
        self,
        path,
        sample_rate: int = 16_000,
        frame_hz: int = 20,
        window_size: float = 20.0,
        stride: float = 20.0,
        max_text_tokens: int = 32,
        cpc_feature_dir: str = "",
    ):
        self.path = path
        self.df = load_df(path)

        self.sample_rate = sample_rate
        self.frame_hz = frame_hz
        self.window_size = window_size
        self.stride = stride
        self.max_text_tokens = max_text_tokens
        self.cpc_feature_dir = cpc_feature_dir

        # ウィンドウをサンプル数とフレーム数に変換
        self.window_samples = int(window_size * sample_rate)
        self.stride_samples = int(stride * sample_rate)
        self.window_frames = int(window_size * frame_hz)
        self.stride_frames = int(stride * frame_hz)

        # キャッシュ（ラベルのみ。CPC特徴量は100Hzで大きいためキャッシュしない）
        self._label_cache: Dict[str, Dict[str, Union[Tensor, List[Tuple[int, int]]]]] = {}

        # 各音声ファイルからのウィンドウインデックスを事前計算
        self._build_window_indices()

    @staticmethod
    def _compute_va_for_window(vad_list, start_frame, n_frames, frame_hz):
        """vad_listからウィンドウ範囲のVAをon-the-flyで計算する。

        入力:
          vad_list: [[start, end], ...] × 2話者のVADセグメント
          start_frame: ウィンドウの開始フレーム
          n_frames: ウィンドウのフレーム数
          frame_hz: フレームレート
        出力: (n_frames, 2) のbinary VAテンソル
        役割: ウィンドウサイズ変更にも柔軟に対応する
        """
        va = torch.zeros(n_frames, 2)
        for speaker_idx, segments in enumerate(vad_list):
            for start_sec, end_sec in segments:
                s = int(start_sec * frame_hz) - start_frame
                e = int(end_sec * frame_hz) - start_frame
                s = max(s, 0)
                e = min(e, n_frames)
                if s < e:
                    va[s:e, speaker_idx] = 1.0
        return va

    def _load_cpc_features(self, session: str) -> Dict[str, Tensor]:
        """事前計算済みCPC特徴量をロードする。

        入力: session - セッションID（例: sw02186）
        出力: feat_1 (n_frames_100hz, 256), feat_2 (n_frames_100hz, 256) を含む辞書
        役割: 100Hz特徴量は1セッション~122MB（float32）のため、キャッシュせず毎回ロードする。
              SSDからの.ptロードは十分高速で、キャッシュするとRAMを圧迫する。
        """
        from pathlib import Path
        cpc_path = Path(self.cpc_feature_dir) / f"{session}_cpc.pt"
        data = torch.load(cpc_path, map_location="cpu", weights_only=True)
        return {
            "feat_1": data["feat_1"].float(),
            "feat_2": data["feat_2"].float(),
        }

    def _load_label(self, label_path: str) -> Dict[str, Union[Tensor, List[Tuple[int, int]]]]:
        """ラベル.ptファイルをロードし、キャッシュする。

        入力: label_path - .ptファイルのパス
        出力: word_tokens, next_filler_class, onset_proximity を含む辞書
        役割: セッション単位でラベルをキャッシュしてI/Oを削減する
        """
        if label_path not in self._label_cache:
            data = torch.load(label_path, map_location="cpu", weights_only=True)
            self._label_cache[label_path] = {
                "word_tokens": data["word_tokens"],  # list of (frame_idx, token_id)
                "next_filler_class": data["next_filler_class"].long(),  # (n_frames,)
                "onset_proximity": data["onset_proximity"].float(),  # (n_frames, 2)
            }
        return self._label_cache[label_path]

    def _build_window_indices(self):
        """各音声ファイルのウィンドウ開始位置をラベルフレーム数に基づいて事前計算する。

        入力: なし（self.df を参照）
        出力: なし（self.window_indices を構築）
        役割: ラベルのフレーム数を基準にウィンドウ分割を決定する
        """
        self.window_indices = []

        for file_idx, d in self.df.iterrows():
            label_path = str(d["label_path"])
            label_data = self._load_label(label_path)
            total_label_frames = label_data["onset_proximity"].shape[0]

            # ラベルフレーム数に基づいてウィンドウを計算
            max_start_frame = total_label_frames - self.window_frames
            if max_start_frame < 0:
                continue

            start_frames = torch.arange(
                0, max_start_frame + 1, self.stride_frames, dtype=torch.long
            )

            for start_frame in start_frames:
                self.window_indices.append(
                    {
                        "file_idx": file_idx,
                        "start_frame": start_frame.item(),
                    }
                )

    def _build_text_context(
        self,
        word_tokens: List[Tuple[int, int]],
        start_frame: int,
        window_frames: int,
    ) -> Tuple[Tensor, Tensor]:
        """ウィンドウに対応するテキストコンテキストを構築する。

        入力:
          word_tokens: (frame_idx, token_id) のリスト（セッション全体、時系列順）
          start_frame: ウィンドウの絶対開始フレーム
          window_frames: ウィンドウのフレーム数
        出力:
          text_tokens: (max_text_tokens,) パディング済みトークンID列
          text_token_positions: (max_text_tokens,) 相対フレーム位置（パディングは-1）
        役割: モデルが各フレームで参照可能な過去の単語コンテキストを構築する。
              相対位置が負の単語はウィンドウ開始前に出現した過去コンテキスト。
              モデルの _gather_text_for_frames は position <= t で判定するため、
              負の位置の単語は常に可視となる（正しい挙動）。
        """
        end_frame = start_frame + window_frames

        # ウィンドウ終了フレームより前の全単語を収集
        relevant = [
            (frame_idx, token_id)
            for frame_idx, token_id in word_tokens
            if frame_idx < end_frame
        ]

        # 直近 max_text_tokens 件を取得
        relevant = relevant[-self.max_text_tokens :]

        n_tokens = len(relevant)
        text_tokens = torch.zeros(self.max_text_tokens, dtype=torch.long)
        text_positions = torch.full((self.max_text_tokens,), -1, dtype=torch.long)

        # 右詰めで格納（パディングが先頭、実トークンが末尾）
        offset = self.max_text_tokens - n_tokens
        for i, (frame_idx, token_id) in enumerate(relevant):
            text_tokens[offset + i] = token_id
            # ウィンドウ内の相対フレーム位置（負の値は過去コンテキスト）
            text_positions[offset + i] = frame_idx - start_frame

        return text_tokens, text_positions

    def __len__(self):
        return len(self.window_indices)

    def clear_cache(self):
        """メモリ使用量削減のためラベルキャッシュをクリアする。"""
        self._label_cache.clear()

    def get_cache_memory_usage(self):
        """キャッシュのメモリ使用量を推定する（デバッグ用）。

        入力: なし
        出力: ラベルキャッシュのメモリ使用量(MB)を含む辞書
        役割: メモリ消費量の監視
        """
        label_memory = 0
        for label_data in self._label_cache.values():
            for key, value in label_data.items():
                if isinstance(value, Tensor):
                    label_memory += value.numel() * value.element_size()
                elif isinstance(value, list):
                    # word_tokens: list of tuples, 概算
                    label_memory += len(value) * 16  # 2 int × 8 bytes

        return {
            "label_cache_mb": label_memory / (1024 * 1024),
            "num_cached_labels": len(self._label_cache),
        }

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, str]]:
        """指定インデックスのウィンドウデータを返す。

        入力: idx - ウィンドウインデックス
        出力: session, waveform, va, s_05, s_10, s_20, next_filler_class,
              text_tokens, text_token_positions, dataset を含む辞書
        役割: 音声・ラベル・テキストコンテキストをウィンドウ単位で切り出して返す
        """
        window_info = self.window_indices[idx]
        file_idx = window_info["file_idx"]
        start_frame = window_info["start_frame"]

        d = self.df.iloc[file_idx]

        # フレーム位置から音声のサンプル位置を計算
        start_time_offset = start_frame / self.frame_hz
        audio_start = float(d["start"]) + start_time_offset
        audio_end = audio_start + self.window_size

        # セッション全体の終了時刻を超えないようクリップ
        audio_end = min(audio_end, float(d["end"]))

        # 音声ロード（ウィンドウ分のみ）
        w_window, _ = load_waveform(
            str(d["audio_path"]),
            start_time=audio_start,
            end_time=audio_end,
            sample_rate=self.sample_rate,
        )

        # ウィンドウサンプル数に合わせてパディングまたはトリム
        if w_window.shape[1] < self.window_samples:
            pad_size = self.window_samples - w_window.shape[1]
            w_window = torch.nn.functional.pad(w_window, (0, pad_size))
        elif w_window.shape[1] > self.window_samples:
            w_window = w_window[:, : self.window_samples]

        w_window = w_window.float()

        # VAをvad_listからon-the-flyで計算
        va_window = self._compute_va_for_window(
            d["vad_list"], start_frame, self.window_frames, self.frame_hz
        )

        # ラベルのウィンドウ切り出し
        label_data = self._load_label(str(d["label_path"]))
        end_frame = start_frame + self.window_frames

        onset_prox_window = label_data["onset_proximity"][start_frame:end_frame].clone()
        next_word_window = label_data["next_filler_class"][start_frame:end_frame].clone()

        # テキストコンテキストの構築
        text_tokens, text_positions = self._build_text_context(
            label_data["word_tokens"],
            start_frame,
            self.window_frames,
        )

        result = {
            "session": str(d["session"]),
            "waveform": w_window,
            "va": va_window,
            "onset_proximity": onset_prox_window,
            "next_filler_class": next_word_window,
            "text_tokens": text_tokens,
            "text_token_positions": text_positions,
            "dataset": str(d["dataset"]),
        }

        # 事前計算済みCPC特徴量があれば追加
        # 特徴量は100Hz（gEncoder+gAR出力）、ラベルは20Hz なのでインデックスを変換する
        if self.cpc_feature_dir:
            cpc_hz = 100
            scale = cpc_hz // self.frame_hz  # 5
            cpc_start = start_frame * scale
            cpc_end = end_frame * scale
            session = str(d["session"])
            cpc_data = self._load_cpc_features(session)
            result["cpc_feat_1"] = cpc_data["feat_1"][cpc_start:cpc_end].clone()
            result["cpc_feat_2"] = cpc_data["feat_2"][cpc_start:cpc_end].clone()

        return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "../data/switchboard/vap_dataset/train.csv"

    # データセットのロード
    dset = VapDataset(path=csv_path)
    print(f"Dataset loaded: {len(dset)} windows")
    print(f"  window_size={dset.window_size}s, stride={dset.stride}s")
    print(f"  window_frames={dset.window_frames}, window_samples={dset.window_samples}")
    print(f"  max_text_tokens={dset.max_text_tokens}")
    print(f"Cache memory: {dset.get_cache_memory_usage()}")
    print()

    # 単一サンプルのテスト
    d = dset[0]
    print("=== Sample 0 shapes ===")
    print(f"  session:              {d['session']}")
    print(f"  dataset:              {d['dataset']}")
    print(f"  waveform:             {d['waveform'].shape}")
    print(f"  va:                   {d['va'].shape}")
    print(f"  s_05:                 {d['s_05'].shape}")
    print(f"  s_10:                 {d['s_10'].shape}")
    print(f"  s_20:                 {d['s_20'].shape}")
    print(f"  next_filler_class:      {d['next_filler_class'].shape}")
    print(f"  text_tokens:          {d['text_tokens'].shape}")
    print(f"  text_token_positions: {d['text_token_positions'].shape}")
    print()

    # テキストコンテキストの内容確認
    non_pad = (d["text_token_positions"] != -1).sum().item()
    print(f"  text context: {non_pad}/{dset.max_text_tokens} tokens filled")
    if non_pad > 0:
        valid_mask = d["text_token_positions"] != -1
        valid_positions = d["text_token_positions"][valid_mask]
        valid_tokens = d["text_tokens"][valid_mask]
        print(f"  position range: [{valid_positions.min().item()}, {valid_positions.max().item()}]")
        print(f"  token ID range: [{valid_tokens.min().item()}, {valid_tokens.max().item()}]")
    print()

    # DataLoaderテスト
    dloader = DataLoader(
        dset,
        batch_size=4,
        num_workers=0,
        shuffle=True,
    )

    batch = next(iter(dloader))
    print("=== Batch shapes ===")
    for key, value in batch.items():
        if isinstance(value, Tensor):
            print(f"  {key:25s} {value.shape}  dtype={value.dtype}")
        else:
            print(f"  {key:25s} {value}")
