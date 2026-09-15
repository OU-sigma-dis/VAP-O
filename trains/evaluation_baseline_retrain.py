"""再学習ベースライン VAP モデルの評価スクリプト。

MaAI の p_now / p_future 方式で評価する（evaluation_baseline.py と同じ方法）。
事前抽出済み CPC 特徴量を使用して高速に推論。

使い方:
  python trains/evaluation_baseline_retrain.py \
    --checkpoint output/checkpoints_baseline_retrain/baseline_retrain-epoch16-val_2.52317.ckpt
"""

import json
import sys
from argparse import ArgumentParser
from os.path import basename, join
from pathlib import Path
from typing import Dict, List, Tuple

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

# VAP-O
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EventConfig
from vap.events import TurnTakingEvents
from utilities.utils import everything_deterministic

# 再学習ベースラインモデル
from train_baseline import BaselineVAPModel
from objective import ObjectiveVAP

everything_deterministic()

MIN_THRESH = 0.01
MAX_THRESH = 0.99
THRESH_STEP = 0.01
ROOT = "runs_evaluation"
FRAME_HZ = 20


def extract_predictions(p_now, p_future, events, batch_size):
    """MaAI の extract_prediction_and_targets と同じ方法で予測値を収集。"""
    results = {"hs": [], "pred_shift": [], "pred_shift_ov": [], "bc": []}
    T = p_now.shape[1]

    for b in range(batch_size):
        # hs: p_now
        if "shift" in events:
            for start, end, speaker in events["shift"][b]:
                if start >= end or start >= T: continue
                end = min(end, T)
                for v in p_now[b, start:end, speaker]:
                    results["hs"].append((v.item(), 1))
        if "hold" in events:
            for start, end, speaker in events["hold"][b]:
                if start >= end or start >= T: continue
                end = min(end, T)
                for v in (1 - p_now[b, start:end, speaker]):
                    results["hs"].append((v.item(), 0))

        # pred_shift: p_future
        if "pred_shift" in events:
            for start, end, speaker in events["pred_shift"][b]:
                if start >= end or start >= T: continue
                end = min(end, T)
                for v in p_future[b, start:end, speaker]:
                    results["pred_shift"].append((v.item(), 1))
        if "pred_shift_neg" in events:
            for start, end, speaker in events["pred_shift_neg"][b]:
                if start >= end or start >= T: continue
                end = min(end, T)
                for v in (1 - p_future[b, start:end, speaker]):
                    results["pred_shift"].append((v.item(), 0))

        # pred_shift_ov: p_future
        if "pred_shift_ov" in events:
            for start, end, speaker in events["pred_shift_ov"][b]:
                if start >= end or start >= T: continue
                end = min(end, T)
                for v in p_future[b, start:end, speaker]:
                    results["pred_shift_ov"].append((v.item(), 1))
        if "pred_shift_ov_neg" in events:
            for start, end, speaker in events["pred_shift_ov_neg"][b]:
                if start >= end or start >= T: continue
                end = min(end, T)
                for v in (1 - p_future[b, start:end, speaker]):
                    results["pred_shift_ov"].append((v.item(), 0))

        # bc: p_now
        if "pred_backchannel" in events:
            for start, end, speaker in events["pred_backchannel"][b]:
                if start >= end or start >= T: continue
                end = min(end, T)
                for v in p_now[b, start:end, speaker]:
                    results["bc"].append((v.item(), 1))
        if "pred_backchannel_neg" in events:
            for start, end, speaker in events["pred_backchannel_neg"][b]:
                if start >= end or start >= T: continue
                end = min(end, T)
                for v in p_now[b, start:end, speaker]:
                    results["bc"].append((v.item(), 0))

    return results


def find_optimal_thresholds(all_preds, all_targets):
    thresholds = np.arange(MIN_THRESH, MAX_THRESH + THRESH_STEP, THRESH_STEP)
    optimal = {}
    for metric in all_preds:
        if not all_preds[metric]:
            optimal[metric] = 0.5
            continue
        preds = np.array(all_preds[metric])
        targets = np.array(all_targets[metric])
        best_f1, best_t = 0, 0.5
        for t in thresholds:
            pb = (preds >= t).astype(float)
            if len(np.unique(pb)) == 1: continue
            f1 = f1_score(targets, pb, average="weighted")
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        optimal[metric] = best_t
        logger.info(f"{metric}: 最適閾値={best_t:.3f} (F1={best_f1:.4f})")
    return optimal


def evaluate_with_thresholds(all_preds, all_targets, thresholds):
    results = {}
    for metric in thresholds:
        if not all_preds.get(metric):
            results[metric] = {"f1": 0, "balanced_accuracy": 0, "precision": 0, "recall": 0,
                               "threshold": thresholds[metric], "n_samples": 0}
            continue
        preds = np.array(all_preds[metric])
        targets = np.array(all_targets[metric])
        t = thresholds[metric]
        pb = (preds >= t).astype(float)
        results[metric] = {
            "f1": f1_score(targets, pb, average="weighted"),
            "balanced_accuracy": balanced_accuracy_score(targets, pb),
            "precision": precision_score(targets, pb, average="weighted", zero_division=0),
            "recall": recall_score(targets, pb, average="weighted", zero_division=0),
            "accuracy": accuracy_score(targets, pb),
            "threshold": t,
            "n_samples": len(targets),
        }
    return results


def collect_predictions(model, dataloader, device, event_extractor):
    objective = ObjectiveVAP(bin_times=[0.2, 0.4, 0.6, 0.8], frame_hz=FRAME_HZ)
    metric_keys = ["hs", "pred_shift", "pred_shift_ov", "bc"]
    all_preds = {k: [] for k in metric_keys}
    all_targets = {k: [] for k in metric_keys}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            for k in batch:
                if isinstance(batch[k], Tensor) and batch[k].is_floating_point():
                    batch[k] = batch[k].to(device).float()
                elif isinstance(batch[k], Tensor):
                    batch[k] = batch[k].to(device)

            # CPC 特徴量から forward
            cpc_feat_1 = batch.get("cpc_feat_1")
            cpc_feat_2 = batch.get("cpc_feat_2")
            if cpc_feat_1 is not None:
                out = model.forward_from_cpc_features(cpc_feat_1, cpc_feat_2)
            else:
                out = model(waveform=batch["waveform"])

            logits = out["logits"]
            probs = logits.softmax(dim=-1)

            # p_now / p_future を CPU で計算
            probs_cpu = probs.cpu()
            p_now = objective.probs_next_speaker_aggregate(probs_cpu, from_bin=0, to_bin=1)
            p_future = objective.probs_next_speaker_aggregate(probs_cpu, from_bin=2, to_bin=3)

            va = batch["va"].cpu()
            events = event_extractor(va)
            B = va.shape[0]

            metric_results = extract_predictions(p_now, p_future, events, B)
            for metric in metric_keys:
                for pred_val, target_val in metric_results[metric]:
                    all_preds[metric].append(pred_val)
                    all_targets[metric].append(target_val)

    return all_preds, all_targets


def evaluate():
    parser = ArgumentParser("Baseline Retrain Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"デバイス: {device}")

    import os
    os.chdir(Path(__file__).resolve().parent.parent)

    name = basename(args.checkpoint).replace(".ckpt", "")
    savepath = join(ROOT, name)
    Path(savepath).mkdir(exist_ok=True, parents=True)

    # モデルロード
    from config import VapConfig
    from config.training import OptConfig, DataConfig
    from model import VapConfig as MaaiVapConfig
    torch.serialization.add_safe_globals([VapConfig, OptConfig, DataConfig, EventConfig,
                                          BaselineVAPModel, MaaiVapConfig])
    model = BaselineVAPModel.load_from_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device).float()
    model.eval()
    logger.info("モデルロード完了")

    event_extractor = TurnTakingEvents(EventConfig())

    # DataLoader
    from datasets.datamodule import VapDataModule
    from config.training import DataConfig
    dconf = DataConfig()
    cpc_dir = str(Path(dconf.train_path).parent / "cpc_features")

    dm = VapDataModule(
        train_path=dconf.train_path,
        val_path=dconf.val_path,
        test_path=dconf.test_path,
        batch_size=16,
        num_workers=4,
        window_size=dconf.window_size,
        stride=dconf.val_stride,
        val_stride=dconf.val_stride,
        frame_hz=FRAME_HZ,
        max_text_tokens=64,
        cpc_feature_dir=cpc_dir,
    )
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")

    # val で閾値探索
    logger.info("=== Step 1: val で閾値探索 ===")
    val_preds, val_targets = collect_predictions(model, dm.val_dataloader(), device, event_extractor)
    optimal_thresholds = find_optimal_thresholds(val_preds, val_targets)
    val_results = evaluate_with_thresholds(val_preds, val_targets, optimal_thresholds)

    for metric, res in val_results.items():
        logger.info(f"  {metric}: F1={res['f1']:.4f}, BalAcc={res['balanced_accuracy']:.4f} "
                     f"(thresh={res['threshold']:.3f}, n={res['n_samples']})")

    # test で評価
    logger.info("=== Step 2: test で評価 ===")
    test_preds, test_targets = collect_predictions(model, dm.test_dataloader(), device, event_extractor)
    test_results = evaluate_with_thresholds(test_preds, test_targets, optimal_thresholds)

    # 結果表示
    print("\n" + "=" * 70)
    print("BASELINE RETRAIN EVALUATION (MaAI architecture, p_now/p_future)")
    print("=" * 70)
    print("\n--- 最適閾値 ---")
    for metric, thresh in optimal_thresholds.items():
        print(f"  {metric}: {thresh:.3f}")
    print(f"\n--- Test イベントメトリクス ---")
    for metric, res in test_results.items():
        print(
            f"  {metric}: F1={res['f1']:.4f}, BalAcc={res['balanced_accuracy']:.4f}, "
            f"Prec={res['precision']:.4f}, Rec={res['recall']:.4f} "
            f"(thresh={res['threshold']:.3f}, n={res['n_samples']})"
        )

    # CSV 保存
    pd.DataFrame([optimal_thresholds]).to_csv(join(savepath, "optimal_thresholds.csv"), index=False)
    test_summary = {}
    for metric, res in test_results.items():
        for k, v in res.items():
            test_summary[f"{metric}_{k}"] = v
    pd.DataFrame([test_summary]).to_csv(join(savepath, "test_results.csv"), index=False)

    print(f"\n結果を保存しました: {savepath}/")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()
