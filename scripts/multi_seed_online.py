"""オンライン評価（疑似オンラインシミュレーション, Table 4）の複数 seed 版。

seed {42,43,44} で VAP-O と baseline を再推論し、online_sim と同一のシミュレーション
（真の onset 非依存, 沈黙中の θ 上方交差で発火）を各 seed で実行して、Table 4 の各行を
seed 間の平均±標準偏差で報告する。fixed_gap はモデル非依存なので全 seed で同一。

使い方: .venv/bin/python scripts/multi_seed_online.py
出力: reports/multi_seed_online.md（seed ごとのキャッシュ reports/_online_cache_<seed>.pkl）
"""
import os
import glob
import pickle
import random
import re
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
from model import VapConfig as MaaiVapConfig
from online_sim import collect, run_system, VAL_CSV, TEST_CSV, CPC_DIR

SEED_NEG = 42
SEEDS = {
    "42": ("output/checkpoints_stereo_onset_seed42/*onset_*.ckpt",
           "output/checkpoints_baseline_retrain/*val_*.ckpt"),
    "43": ("output/checkpoints_stereo_onset_seed43/*onset_*.ckpt",
           "output/checkpoints_baseline_seed43/*val_*.ckpt"),
    "44": ("output/checkpoints_stereo_onset_seed44/*onset_*.ckpt",
           "output/checkpoints_baseline_seed44/*val_*.ckpt"),
}
# 報告する行: (ラベル, system, θ)。θ=None は val F1 最良を各 seed で選択。
ROWS = [
    ("VAP-O extrapolate, val-opt", "vapo_extrap", None),
    ("VAP-O extrapolate, θ=0.8", "vapo_extrap", 0.8),
    ("VAP-O extrapolate, θ=0.9", "vapo_extrap", 0.9),
    ("VAP-O extrapolate, θ=0.95", "vapo_extrap", 0.95),
    ("VAP-O react, θ=0.9", "vapo_react", 0.9),
    ("VAP p_now, val-opt", "pnow_react", None),
    ("VAP p_now, θ=0.9", "pnow_react", 0.9),
    ("Fixed gap", "fixed_gap", 0.0),
]
THETAS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def best_ckpt(pat, key):
    cks = glob.glob(pat)
    return min(cks, key=lambda p: float(re.search(rf'{key}_([0-9.]+)\.ckpt', p).group(1)))


def load_windows(sd, vpat, bpat):
    cache = Path(f"reports/_online_cache_{sd}.pkl")
    if cache.exists():
        d = pickle.load(open(cache, "rb"))
        return d["val"], d["test"]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.serialization.add_safe_globals([VapConfig, OptConfig, DataConfig, EventConfig,
                                          BaselineVAPModel, MaaiVapConfig])
    vapo = VAPModel.load_from_checkpoint(best_ckpt(vpat, "onset"), map_location=dev).to(dev)
    if hasattr(vapo.conf, "text_causal"):
        vapo.conf.text_causal = False
    baseb = BaselineVAPModel.load_from_checkpoint(best_ckpt(bpat, "val"), map_location=dev).to(dev)
    objective = ObjectiveVAP(bin_times=[0.2, 0.4, 0.6, 0.8], frame_hz=20)

    def loader(csv):
        ds = VapDataset(path=csv, window_size=20.0, stride=20.0, cpc_feature_dir=CPC_DIR)
        return DataLoader(ds, batch_size=4, num_workers=0, shuffle=False)

    random.seed(SEED_NEG); np.random.seed(SEED_NEG); torch.manual_seed(SEED_NEG)
    print(f"[{sd}] collect val..."); wv = collect(vapo, baseb, objective, loader(VAL_CSV))
    random.seed(SEED_NEG); np.random.seed(SEED_NEG); torch.manual_seed(SEED_NEG)
    print(f"[{sd}] collect test..."); wt = collect(vapo, baseb, objective, loader(TEST_CSV))
    pickle.dump({"val": wv, "test": wt}, open(cache, "wb"))
    return wv, wt


def select_theta(wins_val, system):
    """val で F1 最良の θ。出力: θ。"""
    best = (THETAS[0], -1)
    for th in THETAS:
        r = run_system(wins_val, system, th)
        if r["f1"] > best[1]:
            best = (th, r["f1"])
    return best[0]


def ms(vals):
    a = np.array(vals, float)
    return a.mean(), a.std(ddof=1)


def main():
    os.chdir(_ROOT)
    # per_seed[row_idx] = list of metric dict over seeds
    per = {i: [] for i in range(len(ROWS))}
    valopt = {i: [] for i in range(len(ROWS))}
    sk = list(SEEDS.keys())
    for sd, (vpat, bpat) in SEEDS.items():
        wv, wt = load_windows(sd, vpat, bpat)
        for i, (_, system, th) in enumerate(ROWS):
            if th is None:
                th_sel = select_theta(wv, system)
                valopt[i].append(th_sel)
            else:
                th_sel = th
            per[i].append(run_system(wt, system, th_sel))
        print(f"[{sd}] done")

    def cell(i, key, pct=False, signed=False):
        vals = [per[i][s][key] for s in range(len(sk))]
        m, sd = ms([v * (100 if pct else 1) for v in vals])
        if pct:
            return f"{m:.1f}$\\pm${sd:.1f}"
        if signed:
            return f"{m:+.2f}$\\pm${sd:.2f}"
        return f"{m:.2f}$\\pm${sd:.2f}"

    L = ["# オンライン評価 複数 seed（seed 42/43/44, Table 4 用）\n",
         "- 各 seed で VAP-O+baseline を再推論し online_sim と同一シミュレーションを実行。",
         "- 各セル mean±std over seeds。fixed gap はモデル非依存（std≈0）。",
         "- 適合率/再現率/±τ は %、バイアス/MAE は秒。\n",
         "| 予測器 | 適合率 | 再現率 | FA/分 | バイアス(s) | MAE(s) | ±0.1s | ±0.3s | ±0.5s |",
         "|---|---|---|---|---|---|---|---|---|"]
    within = lambda i, t: cell(i, "within", pct=True) if False else None
    for i, (label, system, th) in enumerate(ROWS):
        lab = label
        if th is None:
            tset = sorted(set(valopt[i]))
            lab += f"（θ={'/'.join(str(x) for x in tset)}）"
        w = {t: [per[i][s]["within"][t] for s in range(len(sk))] for t in [0.1, 0.3, 0.5]}
        def wcell(t):
            m, s = ms([v * 100 for v in w[t]])
            return f"{m:.1f}$\\pm${s:.1f}"
        L.append(f"| {lab} | {cell(i,'prec',pct=True)} | {cell(i,'rec',pct=True)} | "
                 f"{cell(i,'fa_per_min')} | {cell(i,'bias',signed=True)} | {cell(i,'mae')} | "
                 f"{wcell(0.1)} | {wcell(0.3)} | {wcell(0.5)} |")
    L.append("")
    L.append(f"主要 onset 数(test, 両話者): " +
             ", ".join(f"seed{sd}={per[0][s]['n_major']}" for s, sd in enumerate(sk)))

    out = Path("reports/multi_seed_online.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L)); print("\nsaved ->", out)


if __name__ == "__main__":
    main()
