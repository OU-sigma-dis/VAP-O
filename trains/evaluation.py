"""VAP評価スクリプト。

手順:
  1. valデータで各メトリクス（hs/pred_shift/pred_shift_ov/bc/ls）の最適閾値をF1最大化で探索
  2. testデータでその閾値を使って評価
  3. shift/shift_ov のタイミング精度を評価（2段階: 閾値交差で検出 → ピークでタイミング特定）
  4. フィラーTop-K正解率、S値許容範囲別正解率も算出

使い方:
  cd trains
  python evaluation.py --checkpoint <path_to_ckpt>
  python evaluation.py --checkpoint output/checkpoints/  # ディレクトリ指定で最良ckpt自動選択
"""

from argparse import ArgumentParser
from glob import glob
from os.path import basename, isdir, join
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import Tensor
from tqdm import tqdm
from train import DataConfig, OptConfig, VAPModel, get_run_name

from config import EventConfig, VapConfig
from datasets.datamodule import VapDataModule
from utilities.utils import everything_deterministic
from vap.events import TurnTakingEvents

everything_deterministic()

MIN_THRESH = 0.01
MAX_THRESH = 0.99
THRESH_STEP = 0.01
ROOT = "runs_evaluation"


def collect_predictions(
    model: VAPModel,
    dataloader,
    device: torch.device,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, dict]]:
    """データローダーから全バッチの予測とターゲットを収集する。

    入力: model, dataloader, device
    出力: (all_preds, all_targets, extra) の辞書タプル
      - all_preds/all_targets: 各メトリクスのmax集約された予測値とターゲット
      - extra: filler, s値許容範囲, 損失, タイミングデータ
    """
    metric_keys = ["hs", "pred_shift", "pred_shift_ov", "bc"]
    all_preds = {k: [] for k in metric_keys}
    all_targets = {k: [] for k in metric_keys}

    # イベントメトリクスの生データ保存（MA バリエーション用）
    event_raw_data = []  # [(onset_proximity_batch, events, B), ...]

    # フィラー用
    filler_logits_list = []
    filler_targets_list = []

    # 損失の累積
    total_onset_loss = 0.0
    total_vad_loss = 0.0
    total_filler_loss = 0.0
    num_batches = 0

    # タイミング評価用データ（shift/shift_ov のイベントとモデル出力を保存）
    timing_data = {"shift": [], "shift_ov": []}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting predictions", leave=False):
            # バッチをデバイスに転送
            for key in batch:
                if isinstance(batch[key], Tensor):
                    batch[key] = batch[key].to(device)

            # フォワードパス
            n_frames = batch["onset_proximity"].shape[1]
            out = model(
                audio=batch["waveform"],
                text_tokens=batch["text_tokens"],
                text_token_positions=batch["text_token_positions"],
                n_frames=n_frames,
                cpc_feat_1=batch.get("cpc_feat_1"),
                cpc_feat_2=batch.get("cpc_feat_2"),
            )

            # 損失計算
            t_model = out["onset_proximity"].shape[1]
            t_min = min(t_model, n_frames)

            onset_loss = torch.nn.functional.mse_loss(
                out["onset_proximity"][:, :t_min], batch["onset_proximity"][:, :t_min]
            )
            vad_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out["vad"][:, :t_min], batch["va"][:, :t_min]
            )
            filler_loss = torch.nn.functional.cross_entropy(
                out["filler_logits"][:, :t_min].reshape(-1, model.conf.num_filler_classes),
                batch["next_filler_class"][:, :t_min].reshape(-1),
                ignore_index=0,
            )
            total_onset_loss += onset_loss.item()
            total_vad_loss += vad_loss.item()
            total_filler_loss += filler_loss.item()
            num_batches += 1

            # イベントベースのメトリクス
            if model.event_extractor is not None:
                events = model.event_extractor(batch["va"])
                # raw (MA=1) の結果を即時計算
                preds, targets = model.extract_predictions(
                    out["onset_proximity"], events
                )
                for metric in all_preds.keys():
                    if metric in preds and preds[metric] is not None:
                        all_preds[metric].append(preds[metric].cpu())
                        all_targets[metric].append(targets[metric].cpu())
                # MA バリエーション用に生データを保存
                event_raw_data.append((
                    out["onset_proximity"].cpu(),
                    events,
                    batch["va"].shape[0],
                ))

                # タイミング評価用: pred_shift/pred_shift_ov のイベントと
                # 対応する shift/shift_ov の onset 情報を紐づけて保存
                for pred_key, shift_key in [("pred_shift", "shift"), ("pred_shift_ov", "shift_ov")]:
                    if pred_key not in events or shift_key not in events:
                        continue
                    B = batch["va"].shape[0]
                    for b in range(B):
                        # pred_shift と shift を紐づける（同じ speaker, pred_shift.end ≈ shift.sil_start）
                        for ps_start, ps_end, ps_speaker in events[pred_key][b]:
                            # 対応する shift イベントを探す（pred_shift の end 付近に sil_start がある）
                            best_match = None
                            for sil_start, onset_start, sh_speaker in events[shift_key][b]:
                                if sh_speaker != ps_speaker:
                                    continue
                                # pred_shift の end は沈黙開始付近
                                dist = abs(ps_end - sil_start)
                                if dist <= 5 and (best_match is None or dist < best_match[0]):
                                    best_match = (dist, sil_start, onset_start)
                            if best_match is None:
                                continue
                            _, sil_start, onset_start = best_match
                            true_onset = onset_start if shift_key == "shift" else sil_start
                            timing_data[shift_key].append({
                                "pred_region_start": ps_start,
                                "pred_region_end": ps_end,
                                "sil_start": sil_start,
                                "onset_start": onset_start,
                                "true_onset": true_onset,
                                "speaker": ps_speaker,
                                "onset_proximity": out["onset_proximity"][b, :, ps_speaker].cpu(),
                                "n_frames": t_min,
                            })

            # フィラー
            filler_logits = out["filler_logits"][:, :t_min]
            filler_tgt = batch["next_filler_class"][:, :t_min]
            mask = filler_tgt != 0
            if mask.sum() > 0:
                filler_logits_list.append(filler_logits[mask].cpu())
                filler_targets_list.append(filler_tgt[mask].cpu())

    # 連結
    for metric in all_preds.keys():
        if all_preds[metric]:
            all_preds[metric] = torch.cat(all_preds[metric])
            all_targets[metric] = torch.cat(all_targets[metric])
        else:
            all_preds[metric] = None
            all_targets[metric] = None

    filler_logits_all = torch.cat(filler_logits_list) if filler_logits_list else None
    filler_targets_all = torch.cat(filler_targets_list) if filler_targets_list else None

    extra = {
        "filler_logits": filler_logits_all,
        "filler_targets": filler_targets_all,
        "losses": {
            "onset_loss": total_onset_loss / max(num_batches, 1),
            "vad_loss": total_vad_loss / max(num_batches, 1),
            "filler_loss": total_filler_loss / max(num_batches, 1),
        },
        "timing_data": timing_data,
        "event_raw_data": event_raw_data,
    }

    return all_preds, all_targets, extra


def find_optimal_thresholds(
    all_preds: Dict[str, Tensor],
    all_targets: Dict[str, Tensor],
) -> Dict[str, float]:
    """各メトリクスのF1を最大化する閾値を探索する。"""
    optimal_thresholds = {}
    thresholds = np.arange(MIN_THRESH, MAX_THRESH + THRESH_STEP, THRESH_STEP)

    for metric in all_preds.keys():
        if all_preds[metric] is None or len(all_preds[metric]) == 0:
            logger.warning(f"{metric}: データなし、閾値=0.5をデフォルトに設定")
            optimal_thresholds[metric] = 0.5
            continue

        preds_np = all_preds[metric].numpy()
        targets_np = all_targets[metric].numpy()
        best_f1 = -1
        best_threshold = 0.5

        for thresh in thresholds:
            pred_binary = (preds_np >= thresh).astype(float)
            if len(np.unique(pred_binary)) == 1:
                continue
            f1 = f1_score(targets_np, pred_binary, average="weighted")
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh

        optimal_thresholds[metric] = best_threshold
        logger.info(f"{metric}: 最適閾値={best_threshold:.3f} (F1={best_f1:.4f})")

    return optimal_thresholds


def evaluate_with_thresholds(
    all_preds: Dict[str, Tensor],
    all_targets: Dict[str, Tensor],
    thresholds: Dict[str, float],
) -> Dict[str, dict]:
    """指定した閾値でイベントベースメトリクスを評価する。"""
    results = {}
    for metric in thresholds.keys():
        if all_preds.get(metric) is None or len(all_preds.get(metric, [])) == 0:
            results[metric] = {
                "f1": 0, "balanced_accuracy": 0, "precision": 0, "recall": 0,
                "threshold": thresholds[metric], "n_samples": 0,
            }
            continue

        thresh = thresholds[metric]
        preds_np = all_preds[metric].numpy()
        targets_np = all_targets[metric].numpy()
        pred_binary = (preds_np >= thresh).astype(float)

        results[metric] = {
            "f1": f1_score(targets_np, pred_binary, average="weighted"),
            "balanced_accuracy": balanced_accuracy_score(targets_np, pred_binary),
            "precision": precision_score(targets_np, pred_binary, average="weighted", zero_division=0),
            "recall": recall_score(targets_np, pred_binary, average="weighted", zero_division=0),
            "accuracy": accuracy_score(targets_np, pred_binary),
            "threshold": thresh,
            "n_samples": len(targets_np),
        }

    return results


def _apply_moving_average(signal: Tensor, window: int) -> Tensor:
    """1D信号に因果的な移動平均を適用する。"""
    if window <= 1:
        return signal
    kernel = torch.ones(window, dtype=signal.dtype) / window
    padded = torch.nn.functional.pad(signal, (window - 1, 0))
    return torch.nn.functional.conv1d(
        padded.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0)
    ).squeeze()


def _apply_moving_average_batch(onset_prox: Tensor, window: int) -> Tensor:
    """(B, T, 2) の onset_proximity に因果的な移動平均を適用する。"""
    if window <= 1:
        return onset_prox
    B, T, C = onset_prox.shape
    kernel = torch.ones(1, 1, window, dtype=onset_prox.dtype) / window
    # (B, T, C) → (B*C, 1, T)
    x = onset_prox.permute(0, 2, 1).reshape(B * C, 1, T)
    padded = torch.nn.functional.pad(x, (window - 1, 0))
    smoothed = torch.nn.functional.conv1d(padded, kernel)
    return smoothed.reshape(B, C, T).permute(0, 2, 1)


def evaluate_event_metrics_with_ma(
    val_raw_data: list,
    test_raw_data: list,
    extract_predictions_fn,
    ma_windows: List[int] = None,
) -> Dict[str, dict]:
    """移動平均の複数バリエーションでイベントメトリクスを評価する。

    val で最適閾値を探索し、test で評価。推論は既に完了済みの生データを使用。

    入力:
      val_raw_data: [(onset_proximity, events, B), ...] val の生データ
      test_raw_data: 同 test
      extract_predictions_fn: model.extract_predictions メソッド
      ma_windows: 移動平均の窓サイズリスト（1 は raw と同じなので省略）
    出力: "ma_{window}" → metric → {f1, balanced_accuracy, ...}
    """
    if ma_windows is None:
        ma_windows = [2, 3, 5, 10, 15, 20]

    metric_keys = ["hs", "pred_shift", "pred_shift_ov", "bc"]
    all_results = {}

    for ma_w in ma_windows:
        key = f"ma_{ma_w}"

        # val: 予測値を収集
        val_preds = {k: [] for k in metric_keys}
        val_targets = {k: [] for k in metric_keys}
        for onset_prox, events, B in val_raw_data:
            smoothed = _apply_moving_average_batch(onset_prox, ma_w)
            preds, targets = extract_predictions_fn(smoothed, events)
            for m in metric_keys:
                if m in preds and preds[m] is not None:
                    val_preds[m].append(preds[m].cpu())
                    val_targets[m].append(targets[m].cpu())

        # 連結
        for m in metric_keys:
            if val_preds[m]:
                val_preds[m] = torch.cat(val_preds[m])
                val_targets[m] = torch.cat(val_targets[m])
            else:
                val_preds[m] = None
                val_targets[m] = None

        # val で閾値探索
        thresholds = find_optimal_thresholds(val_preds, val_targets)

        # test: 予測値を収集
        test_preds = {k: [] for k in metric_keys}
        test_targets = {k: [] for k in metric_keys}
        for onset_prox, events, B in test_raw_data:
            smoothed = _apply_moving_average_batch(onset_prox, ma_w)
            preds, targets = extract_predictions_fn(smoothed, events)
            for m in metric_keys:
                if m in preds and preds[m] is not None:
                    test_preds[m].append(preds[m].cpu())
                    test_targets[m].append(targets[m].cpu())

        for m in metric_keys:
            if test_preds[m]:
                test_preds[m] = torch.cat(test_preds[m])
                test_targets[m] = torch.cat(test_targets[m])
            else:
                test_preds[m] = None
                test_targets[m] = None

        # test で評価
        all_results[key] = evaluate_with_thresholds(test_preds, test_targets, thresholds)

    return all_results


def _evaluate_timing_single(
    timing_data_events: list,
    detect_threshold: float,
    frame_hz: int,
    ma_window: int = 1,
) -> dict:
    """単一の移動平均窓サイズでタイミング精度を評価する。"""
    tolerances = [0.05, 0.1, 0.3, 0.5, 1.0]
    errors = []
    detected_count = 0

    for ev in timing_data_events:
        true_onset = ev["true_onset"]
        n_frames = ev["n_frames"]
        pred_start = ev["pred_region_start"]
        pred_end = ev["pred_region_end"]

        if true_onset >= n_frames:
            continue

        signal = ev["onset_proximity"][:n_frames]
        signal = _apply_moving_average(signal, ma_window)

        # Step 1: pred_shift 区間内で閾値交差を探す
        crossing_abs = None
        seg = signal[pred_start:pred_end]
        if len(seg) == 0:
            errors.append(None)
            continue

        if seg[0] >= detect_threshold:
            crossing_abs = pred_start
        else:
            for i in range(1, len(seg)):
                if seg[i] >= detect_threshold and seg[i - 1] < detect_threshold:
                    crossing_abs = pred_start + i
                    break

        if crossing_abs is None:
            errors.append(None)
            continue

        detected_count += 1

        # Step 2: 検出位置以降の大域ピーク（argmax）。
        # onset proximity は onset で最大になるよう設計されているため、最初の局所ピーク
        # ではなく探索窓内の大域ピークが真の onset に最もよく整合する（読み出し改善）。
        search_end = min(n_frames, true_onset + int(1.0 * frame_hz))
        seg2_end = max(crossing_abs + 1, search_end)
        peak = crossing_abs + int(np.argmax(signal[crossing_abs:seg2_end]))

        predicted_onset = peak + 1
        error_sec = (predicted_onset - true_onset) / frame_hz
        errors.append(error_sec)

    total = len(errors)
    detected_errors = [e for e in errors if e is not None]
    detected_np = np.array(detected_errors) if detected_errors else np.array([])

    result = {
        "n_total": total,
        "n_detected": detected_count,
        "detection_rate": detected_count / total if total > 0 else 0,
    }

    if len(detected_np) > 0:
        result["mean_error"] = float(detected_np.mean())
        result["abs_mean_error"] = float(np.abs(detected_np).mean())
        result["median_error"] = float(np.median(detected_np))
        for tol in tolerances:
            within = (np.abs(detected_np) <= tol).sum()
            result[f"within_{tol}s"] = within / total

    return result


def _find_timing_threshold(
    timing_data_events: list,
    frame_hz: int,
    ma_window: int = 1,
) -> float:
    """val データからタイミング検出の最適閾値を探索する（検出率×±0.5s正解率 最大化）。"""
    best_score = -1
    best_t = 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        r = _evaluate_timing_single(timing_data_events, t, frame_hz, ma_window)
        det = r.get("detection_rate", 0)
        w05 = r.get("within_0.5s", 0)
        score = det * w05
        if score > best_score:
            best_score = score
            best_t = t
    return best_t


def evaluate_timing(
    val_timing_data: Dict[str, list],
    test_timing_data: Dict[str, list],
    frame_hz: int = 20,
    ma_windows: List[int] = None,
) -> Dict[str, dict]:
    """移動平均の複数バリエーションでタイミング精度を評価する。

    val データで最適閾値を探索し、test データで評価する。

    入力:
      val_timing_data: val のタイミングデータ
      test_timing_data: test のタイミングデータ
      frame_hz: フレームレート
      ma_windows: 移動平均の窓サイズリスト
    出力: "ma_{window}" → event_type → timing結果 の辞書
    """
    if ma_windows is None:
        ma_windows = [1, 2, 3, 5, 10, 15, 20]

    all_results = {}

    for ma_w in ma_windows:
        key = f"ma_{ma_w}" if ma_w > 1 else "raw"
        all_results[key] = {}

        for event_type in ["shift", "shift_ov"]:
            val_events = val_timing_data.get(event_type, [])
            test_events = test_timing_data.get(event_type, [])

            if not test_events:
                all_results[key][event_type] = {"n_total": 0}
                continue

            # val で最適閾値を探索
            if val_events:
                thresh = _find_timing_threshold(val_events, frame_hz, ma_w)
            else:
                thresh = 0.5

            # test で評価
            result = _evaluate_timing_single(test_events, thresh, frame_hz, ma_w)
            result["threshold"] = thresh
            all_results[key][event_type] = result

    return all_results


def compute_filler_topk(logits: Tensor, targets: Tensor) -> Dict[str, float]:
    """フィラーTop-K正解率を計算する。"""
    if logits is None or targets is None:
        return {"top1": 0.0, "top3": 0.0}

    top1_correct = (logits.argmax(dim=-1) == targets).float().mean().item()

    top3_preds = logits.topk(3, dim=-1).indices
    top3_correct = (top3_preds == targets.unsqueeze(-1)).any(dim=-1).float().mean().item()

    return {"top1": top1_correct, "top3": top3_correct}


def get_args():
    parser = ArgumentParser("VAP Evaluation")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="チェックポイントファイルまたはディレクトリ",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="デバイス（auto/mps/cuda/cpu）",
    )
    parser.add_argument(
        "--text_causal",
        type=str,
        choices=["auto", "true", "false"],
        default="auto",
        help=(
            "テキスト Cross-Attention を causal にするか。"
            "auto: パス名に 'causal' を含むckptのみ causal、それ以外は非causal（旧ckptは"
            "マスク未実装で学習されているため既定で非causal に強制）。"
            "true/false: 明示的に上書き。"
        ),
    )
    return parser.parse_args()


def resolve_checkpoint(checkpoint_path: str) -> str:
    """チェックポイントパスを解決する。ディレクトリの場合は最良のckptを自動選択。"""
    if not isdir(checkpoint_path):
        return checkpoint_path

    checkpoints = glob(join(checkpoint_path, "*.ckpt"))
    if not checkpoints:
        raise ValueError(f"ディレクトリにckptファイルが見つかりません: {checkpoint_path}")

    best_ckpt = None
    best_val_loss = float("inf")
    for c in checkpoints:
        try:
            val_loss = float(c.split("val_")[1].split(".ckpt")[0])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = c
        except (IndexError, ValueError):
            continue

    if best_ckpt is None:
        best_ckpt = checkpoints[0]

    logger.info(f"選択されたチェックポイント: {best_ckpt}")
    return best_ckpt


def get_device(device_str: str) -> torch.device:
    """デバイスを解決する。"""
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def evaluate():
    args = get_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    device = get_device(args.device)
    logger.info(f"デバイス: {device}")

    # プロジェクトルートに移動（相対パスを解決するため）
    import os
    os.chdir(Path(__file__).resolve().parent.parent)

    # 保存先ディレクトリ
    name = basename(checkpoint).replace(".ckpt", "")
    savepath = join(ROOT, name)
    Path(savepath).mkdir(exist_ok=True, parents=True)

    # モデルロード
    torch.serialization.add_safe_globals([VapConfig, OptConfig, DataConfig, EventConfig])
    model = VAPModel.load_from_checkpoint(checkpoint)

    # --- ガード: text_causal を学習時の設定に一致させる ---
    # 注意: text_causal フィールドを後から追加したため、旧ckpt（マスク未実装で
    #       非causal学習）をロードすると dataclass 既定の True が後付けされ、
    #       そのまま評価すると「非causal学習なのにcausalマスク」で数値が壊れる。
    #       auto では、パスに 'causal' を含むckptのみ causal とし、それ以外は
    #       非causal に強制する（既存ckptはすべて非causalで学習されているため安全）。
    if getattr(model, "conf", None) is not None and hasattr(model.conf, "text_causal"):
        if args.text_causal == "true":
            resolved = True
        elif args.text_causal == "false":
            resolved = False
        else:  # auto
            resolved = "causal" in basename(checkpoint).lower() or "causal" in checkpoint.lower()
        if model.conf.text_causal != resolved:
            logger.warning(
                f"text_causal を {model.conf.text_causal} -> {resolved} に上書き "
                f"(--text_causal={args.text_causal}, ckpt='{basename(checkpoint)}')"
            )
        model.conf.text_causal = resolved
        logger.info(f"評価時 text_causal = {model.conf.text_causal}")

    model.event_extractor = TurnTakingEvents(EventConfig())
    model = model.to(device)
    model.eval()
    logger.info("モデルロード完了")

    # データロード
    dconf = DataConfig()
    cpc_feature_dir = str(Path(dconf.train_path).parent / "cpc_features")

    dm = VapDataModule(
        train_path=dconf.train_path,
        val_path=dconf.val_path,
        test_path=dconf.test_path,
        batch_size=dconf.batch_size,
        num_workers=dconf.num_workers,
        window_size=dconf.window_size,
        stride=dconf.stride,
        val_stride=dconf.val_stride,
        frame_hz=model.conf.frame_hz,
        max_text_tokens=model.conf.max_text_tokens,
        cpc_feature_dir=cpc_feature_dir,
    )
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")

    # ============================
    # Step 1: valデータで最適閾値探索
    # ============================
    logger.info("=== Step 1: valデータで最適閾値を探索 ===")
    val_preds, val_targets, val_extra = collect_predictions(
        model, dm.val_dataloader(), device
    )

    optimal_thresholds = find_optimal_thresholds(val_preds, val_targets)

    # val結果も表示
    val_results = evaluate_with_thresholds(val_preds, val_targets, optimal_thresholds)
    val_filler = compute_filler_topk(val_extra["filler_logits"], val_extra["filler_targets"])

    logger.info("--- Val結果（最適閾値） ---")
    logger.info(f"  Loss: S={val_extra['losses']['onset_loss']:.4f}, "
                f"VAD={val_extra['losses']['vad_loss']:.4f}, "
                f"Filler={val_extra['losses']['filler_loss']:.4f}")
    for metric, res in val_results.items():
        logger.info(f"  {metric}: F1={res['f1']:.4f}, BalAcc={res['balanced_accuracy']:.4f}, "
                     f"Prec={res['precision']:.4f}, Rec={res['recall']:.4f} "
                     f"(thresh={res['threshold']:.3f}, n={res['n_samples']})")
    logger.info(f"  Filler: Top1={val_filler['top1']:.4f}, Top3={val_filler['top3']:.4f}")

    # ============================
    # Step 2: testデータで評価
    # ============================
    logger.info("=== Step 2: testデータで最適閾値を使って評価 ===")
    test_preds, test_targets, test_extra = collect_predictions(
        model, dm.test_dataloader(), device
    )

    test_results = evaluate_with_thresholds(test_preds, test_targets, optimal_thresholds)
    test_filler = compute_filler_topk(test_extra["filler_logits"], test_extra["filler_targets"])

    # ============================
    # Step 3: タイミング精度評価（移動平均バリエーション）
    # ============================
    logger.info("=== Step 3: タイミング精度評価 ===")
    test_timing = evaluate_timing(
        val_timing_data=val_extra["timing_data"],
        test_timing_data=test_extra["timing_data"],
        frame_hz=model.conf.frame_hz,
        ma_windows=[1, 2, 3, 5, 10, 15, 20],
    )

    # ============================
    # Step 4: イベントメトリクス MA バリエーション
    # ============================
    logger.info("=== Step 4: イベントメトリクス MA バリエーション ===")
    event_ma_results = evaluate_event_metrics_with_ma(
        val_raw_data=val_extra["event_raw_data"],
        test_raw_data=test_extra["event_raw_data"],
        extract_predictions_fn=model.extract_predictions,
        ma_windows=[2, 3, 5, 10, 15, 20],
    )

    # ============================
    # 結果表示
    # ============================
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print("\n--- 最適閾値（valデータで探索） ---")
    for metric, thresh in optimal_thresholds.items():
        print(f"  {metric}: {thresh:.3f}")

    print(f"\n--- Test Loss ---")
    print(f"  S: {test_extra['losses']['onset_loss']:.4f}")
    print(f"  VAD: {test_extra['losses']['vad_loss']:.4f}")
    print(f"  Filler: {test_extra['losses']['filler_loss']:.4f}")

    print(f"\n--- Test イベントメトリクス ---")
    print("  [raw]")
    for metric, res in test_results.items():
        print(
            f"    {metric}: F1={res['f1']:.4f}, BalAcc={res['balanced_accuracy']:.4f}, "
            f"Prec={res['precision']:.4f}, Rec={res['recall']:.4f} "
            f"(thresh={res['threshold']:.3f}, n={res['n_samples']})"
        )
    for ma_key, ma_res in event_ma_results.items():
        print(f"  [{ma_key}]")
        for metric, res in ma_res.items():
            if res.get("n_samples", 0) == 0:
                continue
            print(
                f"    {metric}: F1={res['f1']:.4f}, BalAcc={res['balanced_accuracy']:.4f} "
                f"(thresh={res['threshold']:.3f})"
            )

    print(f"\n--- Test タイミング精度（pred_shift区間→ピーク検出） ---")
    for ma_key, ma_results in test_timing.items():
        print(f"\n  [{ma_key}]")
        for event_type in ["shift", "shift_ov"]:
            tr = ma_results.get(event_type, {})
            n = tr.get("n_total", 0)
            if n == 0:
                continue
            det_rate = tr.get("detection_rate", 0)
            thresh = tr.get("threshold", 0)
            line = f"    {event_type}: n={n}, 検出率={det_rate:.3f}, thresh={thresh:.2f}"
            if "abs_mean_error" in tr:
                line += f", |err|={tr['abs_mean_error']:.3f}s"
                for tol in [0.1, 0.3, 0.5]:
                    k = f"within_{tol}s"
                    if k in tr:
                        line += f", ±{tol}s={tr[k]:.3f}"
            print(line)

    print(f"\n--- Test フィラー正解率 ---")
    print(f"  Top-1: {test_filler['top1']:.4f}")
    print(f"  Top-3: {test_filler['top3']:.4f}")

    # ============================
    # CSV保存
    # ============================
    # 最適閾値
    pd.DataFrame([optimal_thresholds]).to_csv(join(savepath, "optimal_thresholds.csv"), index=False)

    # テスト結果
    test_summary = {}
    test_summary.update(test_extra["losses"])
    for metric, res in test_results.items():
        for k, v in res.items():
            test_summary[f"{metric}_{k}"] = v
    test_summary["filler_top1"] = test_filler["top1"]
    test_summary["filler_top3"] = test_filler["top3"]
    # タイミング結果（各移動平均バリエーション）
    for ma_key, ma_results in test_timing.items():
        for event_type in ["shift", "shift_ov"]:
            tr = ma_results.get(event_type, {})
            for k, v in tr.items():
                test_summary[f"timing_{ma_key}_{event_type}_{k}"] = v

    # イベント MA バリエーション結果
    for ma_key, ma_res in event_ma_results.items():
        for metric, res in ma_res.items():
            for k, v in res.items():
                test_summary[f"event_{ma_key}_{metric}_{k}"] = v

    pd.DataFrame([test_summary]).to_csv(join(savepath, "test_results.csv"), index=False)

    print(f"\n結果を保存しました: {savepath}/")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()
