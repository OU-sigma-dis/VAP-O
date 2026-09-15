"""複数 seed 検証: 正準プロトコル（final_stats と同一）で全 seed を一括評価し mean±std を算出。

役割: 査読の複数 seed 要求への回答。ベースライン {42,43,44} と VAP-O {42,43,44}（+参考 seed1）を
      seed ごとにペアで評価する。各 seed で final_stats.py と同一の手続き:
      - 同一バッチ・同一イベント集合で両モデルを評価（random.seed(42) で負例固定）
      - 検出: VAP-O=イベント単位（集約とmax/lastとしきい値を val 選択）,
              ベースライン=原著フレーム単位（しきい値 val 選択）
      - タイミング: VAP-O は readout(argmax固定+peak候補)/MA/thr を val 選択（被覆込み±0.3s）。
        ベースラインの反応的 p_now 読み出し（原著, val選択しきい値）も併算。
      - 沈黙経由のみ（重複経由のオフライン評価は退化のため対象外）。
      集計: seed {42,43,44} の mean±std（seed1 は参考行）。既存 seed の値は final_stats の
      再現であることを確認できる。

使い方: .venv/bin/python scripts/multi_seed_eval.py
出力: reports/multi_seed.md（seed ごとのキャッシュ reports/_mseed_cache_<tag>.pkl）
"""
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "trains"))
sys.path.insert(0, str(_ROOT / "vap")); sys.path.insert(0, str(_ROOT / "scripts"))

from config import EventConfig, VapConfig, OptConfig
from config.training import DataConfig
from datasets.dataset import VapDataset
from objective import ObjectiveVAP
from train import VAPModel
from train_baseline import BaselineVAPModel
from vap.events import TurnTakingEvents
from model import VapConfig as MaaiVapConfig
from tier1b_paired import run_models, localize_argmax, localize_firstpeak
from final_stats import (agg, wf1, best_thr, timing_summary, vo_errors, AGG_CANDS,
                         onset_errors, FRAME_HZ, TOLS)
from utilities.paths import switchboard_paths

SEED = 42
_, VAL_CSV, TEST_CSV, CPC_DIR = switchboard_paths()
METRIC_KEYS = ["hs", "pred_shift", "pred_shift_ov", "bc"]

import glob
import re


def best_ckpt(pattern, key):
    """ディレクトリから最良チェックポイントを選ぶ。key='onset' or 'val'。"""
    cks = glob.glob(pattern)
    assert cks, f"no ckpt: {pattern}"
    return min(cks, key=lambda p: float(re.search(rf'{key}_([0-9.]+)\.ckpt', p).group(1)))


PAIRS = {  # seed_tag -> (vapo_ckpt, base_ckpt)
    "42": (best_ckpt("output/checkpoints_stereo_onset_seed42/*onset_*.ckpt", "onset"),
           best_ckpt("output/checkpoints_baseline_retrain/*val_*.ckpt", "val")),
    "43": (best_ckpt("output/checkpoints_stereo_onset_seed43/*onset_*.ckpt", "onset"),
           best_ckpt("output/checkpoints_baseline_seed43/*val_*.ckpt", "val")),
    "44": (best_ckpt("output/checkpoints_stereo_onset_seed44/*onset_*.ckpt", "onset"),
           best_ckpt("output/checkpoints_baseline_seed44/*val_*.ckpt", "val")),
    "1(参考)": (best_ckpt("output/checkpoints_stereo_onset/*onset_*.ckpt", "onset"),
                best_ckpt("output/checkpoints_baseline_retrain/*val_*.ckpt", "val")),
}
# 注: baseline の既存学習は pl.seed_everything(42) ハードコードだったため seed42 に対応。
# VAP-O seed1(参考) 行のベースライン側は seed42 と同一（参考行では VAP-O 側のみ意味を持つ）。


def collect_pair(tag, vapo_ckpt, base_ckpt):
    """1 seed ペアの val/test 収集（final_stats と同一手続き）。キャッシュ付き。"""
    cache = Path(f"reports/_mseed_cache_{tag.replace('(参考)','ref')}.pkl")
    if cache.exists():
        d = pickle.load(open(cache, "rb"))
        return d["data"]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.serialization.add_safe_globals([VapConfig, OptConfig, DataConfig, EventConfig,
                                          BaselineVAPModel, MaaiVapConfig])
    vapo = VAPModel.load_from_checkpoint(vapo_ckpt, map_location=dev).to(dev)
    if hasattr(vapo.conf, "text_causal"):
        vapo.conf.text_causal = False
    baseb = BaselineVAPModel.load_from_checkpoint(base_ckpt, map_location=dev).to(dev)
    objective = ObjectiveVAP(bin_times=[0.2, 0.4, 0.6, 0.8], frame_hz=FRAME_HZ)

    def loader(csv):
        d_ = VapDataset(path=csv, window_size=20.0, stride=20.0, cpc_feature_dir=CPC_DIR)
        return DataLoader(d_, batch_size=4, num_workers=0, shuffle=False)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"[{tag}] collect val...")
    dv, tv = run_models(vapo, baseb, objective, loader(VAL_CSV), TurnTakingEvents(EventConfig()))
    print(f"[{tag}] collect test...")
    dt, tt = run_models(vapo, baseb, objective, loader(TEST_CSV), TurnTakingEvents(EventConfig()))
    pickle.dump({"data": (dv, tv, dt, tt),
                 "meta": {"vapo": vapo_ckpt, "base": base_ckpt}}, open(cache, "wb"))
    return dv, tv, dt, tt


def eval_pair(dv, tv, dt, tt):
    """1 seed ペアの指標算出（final_stats と同一の val 選択）。出力: dict。"""
    out = {}
    grid_v = np.arange(0.0, 1.0001, 0.005)
    grid_b = np.arange(0.01, 0.991, 0.01)
    for m in METRIC_KEYS:
        # VAP-O: 集約+しきい値を val 選択
        best_a, best_f, best_t = None, -1.0, None
        for a in AGG_CANDS:
            vs = [agg(x[0], a) for x in dv[m]]
            vt = [x[2] for x in dv[m]]
            t_, f_ = best_thr(vs, vt, grid_v)
            if f_ > best_f:
                best_a, best_f, best_t = a, f_, t_
        ts = np.array([agg(x[0], best_a) for x in dt[m]])
        yt = np.array([x[2] for x in dt[m]])
        out[f"vapo_{m}"] = wf1(yt, (ts >= best_t).astype(int))
        # baseline: 原著フレーム単位
        fv = np.concatenate([x[1] for x in dv[m]])
        fvt = np.concatenate([np.full(len(x[1]), x[2]) for x in dv[m]])
        bt_, _ = best_thr(fv, fvt, grid_b)
        ft = np.concatenate([x[1] for x in dt[m]])
        ftt = np.concatenate([np.full(len(x[1]), x[2]) for x in dt[m]])
        out[f"base_{m}"] = wf1(ftt, (ft >= bt_).astype(int))

    # タイミング（沈黙経由のみ, VAP-O: readout/MA/thr を val 選択）
    best = None
    for peak in ["argmax", "first_peak"]:
        for w in [1, 5, 10]:
            for thr in np.round(np.arange(0.15, 0.71, 0.05), 2):
                ae = vo_errors(tv["shift"], thr, w, peak)
                _, _, within = timing_summary(ae)
                sc = within[0.3]
                if best is None or sc > best[0]:
                    best = (sc, peak, w, thr)
    _, peak, w, thr = best
    ae = vo_errors(tt["shift"], thr, w, peak)
    det, mae, within = timing_summary(ae)
    out["vapo_det"] = det
    out["vapo_mae"] = mae
    for t in TOLS:
        out[f"vapo_w{t}"] = within[t]
    out["vapo_timing_cfg"] = f"{peak}/MA{w}/thr{thr}"

    # baseline p_now 反応的読み出し（原著）
    bthr, bsc = 0.3, -1
    for thr_ in np.round(np.arange(0.30, 0.71, 0.05), 2):
        ae_ = onset_errors(tv["shift"], thr_)
        _, _, wv = timing_summary(ae_)
        if wv[0.5] > bsc:
            bsc, bthr = wv[0.5], thr_
    ae_b = onset_errors(tt["shift"], bthr)
    detb, maeb, wb = timing_summary(ae_b)
    out["base_mae"] = maeb
    out["base_w0.1"] = wb[0.1]
    return out


def ms(vals):
    """mean±std 文字列（seed {42,43,44} 用）。"""
    a = np.array(vals, dtype=float)
    return f"{a.mean():.3f}±{a.std(ddof=1):.3f}"


def main():
    os.chdir(_ROOT)
    results = {}
    for tag, (vc, bc) in PAIRS.items():
        print(f"== {tag}: vapo={Path(vc).name} base={Path(bc).name}")
        dv, tv, dt, tt = collect_pair(tag, vc, bc)
        results[tag] = eval_pair(dv, tv, dt, tt)
        print(results[tag])

    main_seeds = ["42", "43", "44"]
    L = ["# 複数 seed 検証（seed 42/43/44, 正準プロトコル）\n",
         "- 各 seed で final_stats と同一の評価（同一イベント・val 選択・被覆込み±τ）。",
         "- 検出: VAP-O=イベント単位読み出し / ベースライン=原著フレーム単位。",
         "- タイミング: 沈黙経由のみ（重複経由のオフライン評価は退化のため対象外）。",
         "- VAP-O seed1 は開発時の元モデル（参考行）。\n"]

    L.append("## 検出 F1（test）")
    hdr = "| 指標 | " + " | ".join(f"seed{t}" for t in main_seeds) + " | mean±std | seed1(参考) |"
    L.append(hdr); L.append("|" + "---|" * (len(main_seeds) + 3))
    names = {"hs": "hs", "pred_shift": "shift(沈黙)", "pred_shift_ov": "shift(重複)", "bc": "bc"}
    for m in METRIC_KEYS:
        for model, label in [("vapo", "VAP-O"), ("base", "Baseline")]:
            vals = [results[t][f"{model}_{m}"] for t in main_seeds]
            ref = results["1(参考)"][f"{model}_{m}"]
            L.append(f"| {label} {names[m]} | " +
                     " | ".join(f"{v:.3f}" for v in vals) +
                     f" | **{ms(vals)}** | {ref:.3f} |")
    L.append("")
    L.append("## onset タイミング（沈黙経由, test, 被覆込み）")
    L.append(hdr); L.append("|" + "---|" * (len(main_seeds) + 3))
    rows = [("VAP-O MAE(s)", "vapo_mae"), ("VAP-O ±0.1s", "vapo_w0.1"),
            ("VAP-O ±0.3s", "vapo_w0.3"), ("VAP-O ±0.5s", "vapo_w0.5"),
            ("VAP-O det", "vapo_det"),
            ("Baseline p_now MAE(s)", "base_mae"), ("Baseline p_now ±0.1s", "base_w0.1")]
    for label, key in rows:
        vals = [results[t][key] for t in main_seeds]
        ref = results["1(参考)"][key]
        L.append(f"| {label} | " + " | ".join(f"{v:.3f}" for v in vals) +
                 f" | **{ms(vals)}** | {ref:.3f} |")
    L.append("")
    L.append("採用タイミング設定(val選択): " +
             ", ".join(f"seed{t}={results[t]['vapo_timing_cfg']}" for t in main_seeds))

    out = Path("reports/multi_seed.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L)); print("\nsaved ->", out)


if __name__ == "__main__":
    main()
