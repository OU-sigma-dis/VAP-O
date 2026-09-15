"""複数 seed での対応のある統計検定（査読対応: 統計検定も multi-seed で取り直す）。

各 seed {42,43,44} で VAP-O と baseline を同一イベント上で再推論し、
VAP-O の onset proximity と baseline の 256 クラス由来信号（p_now/p_future/bin 期待値）を
1 パスで整合収集する。seed ごとに final_stats/vap_timing_readouts と同一の val 選択で
per-event 誤差を求め、VAP-O vs {p_now(反応的), 固定ギャップ, p_future+δ} の対応検定
（exact McNemar ±τ Holm 補正, Wilcoxon, Cohen's h/dz）を実施。最後に 3 seed を集約し、
点推定は mean±std、有意性は「全 seed で有意（最保守 Holm p）」として報告する。沈黙経由のみ。

使い方: .venv/bin/python scripts/multi_seed_stats.py
出力: reports/multi_seed_stats.md（seed ごとのキャッシュ reports/_msstats_<seed>.pkl）
"""
import os
import pickle
import random
import re
import sys
import glob
from pathlib import Path

import numpy as np
import torch
from scipy import stats
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
from tier1b_paired import localize_argmax, localize_firstpeak, moving_average
from vap_timing_readouts import predict_event, crossing_in_region, BIN_CENTERS
from utilities.paths import switchboard_paths

FRAME_HZ = 20
TOLS = [0.1, 0.3, 0.5, 1.0]
SEED_NEG = 42  # 負例サンプリング固定（イベント集合を全 seed で共通化）
_, VAL_CSV, TEST_CSV, CPC_DIR = switchboard_paths()

SEEDS = {
    "42": ("output/checkpoints_stereo_onset_seed42/*onset_*.ckpt",
           "output/checkpoints_baseline_retrain/*val_*.ckpt"),
    "43": ("output/checkpoints_stereo_onset_seed43/*onset_*.ckpt",
           "output/checkpoints_baseline_seed43/*val_*.ckpt"),
    "44": ("output/checkpoints_stereo_onset_seed44/*onset_*.ckpt",
           "output/checkpoints_baseline_seed44/*val_*.ckpt"),
}


def best_ckpt(pat, key):
    cks = glob.glob(pat)
    return min(cks, key=lambda p: float(re.search(rf'{key}_([0-9.]+)\.ckpt', p).group(1)))


def collect_both(vapo, baseb, objective, loader):
    """1 パスで VAP-O vo と baseline 256 クラス由来信号を shift イベント単位に整合収集。
    出力: [dict(vo, pnow, pfut, q(4), pf5(5), ps, pe, onset)]（沈黙経由 shift のみ）。"""
    idx = torch.arange(256)
    states = objective.codebook.decode(idx)  # (256,2,4)
    onehot_first = torch.zeros(256, 2, 5)
    for c in range(256):
        for s in range(2):
            nz = torch.nonzero(states[c, s])
            onehot_first[c, s, int(nz[0]) if len(nz) else 4] = 1.0
    ev_out = []
    dev = next(vapo.parameters()).device
    vapo.eval(); baseb.eval()
    ee = TurnTakingEvents(EventConfig())
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(dev)
                    if batch[k].is_floating_point():
                        batch[k] = batch[k].float()
            nf = batch["onset_proximity"].shape[1]
            vo = vapo(audio=batch["waveform"], text_tokens=batch["text_tokens"],
                      text_token_positions=batch["text_token_positions"], n_frames=nf,
                      cpc_feat_1=batch.get("cpc_feat_1"), cpc_feat_2=batch.get("cpc_feat_2"))
            op = vo["onset_proximity"].cpu().numpy()
            bo = baseb.forward_from_cpc_features(batch.get("cpc_feat_1"), batch.get("cpc_feat_2"))
            probs = bo["logits"].softmax(dim=-1).cpu()
            q = torch.einsum("btc,csk->btsk", probs, states.float()).numpy()
            pf5 = torch.einsum("btc,csj->btsj", probs, onehot_first).numpy()
            pnow = objective.probs_next_speaker_aggregate(probs, 0, 1).numpy()
            pfut = objective.probs_next_speaker_aggregate(probs, 2, 3).numpy()
            va = batch["va"].cpu()
            ev = ee(va)
            B, T = op.shape[0], op.shape[1]
            for b in range(B):
                if "pred_shift" not in ev or "shift" not in ev:
                    continue
                for ps, pe, spk in ev["pred_shift"][b]:
                    best = None
                    for a, o, sp2 in ev["shift"][b]:
                        if sp2 != spk:
                            continue
                        d = abs(pe - a)
                        if d <= 5 and (best is None or d < best[0]):
                            best = (d, a, o)
                    if best is None:
                        continue
                    _, sil, ons = best
                    if not (0 <= ps < pe <= T and 0 <= ons < T):
                        continue
                    ev_out.append(dict(
                        vo=op[b, :, spk].astype(np.float32),
                        pnow=pnow[b, :, spk].astype(np.float32),
                        pfut=pfut[b, :, spk].astype(np.float32),
                        q=q[b, :, spk].astype(np.float32),
                        pf5=pf5[b, :, spk].astype(np.float32),
                        ps=int(ps), pe=int(pe), onset=int(ons)))
    return ev_out


def summ(errs):
    ae = np.abs(np.array([e if e is not None else np.nan for e in errs]))
    det = float(np.mean(~np.isnan(ae)))
    mae = float(np.nanmean(ae)) if det > 0 else np.nan
    within = {t: float((np.nan_to_num(ae, nan=1e9) <= t).mean()) for t in TOLS}
    return det, mae, within, ae


def vapo_errs(events, thr, w, peak):
    out = []
    for e in events:
        if peak == "argmax":
            po = localize_argmax(e["vo"], e["ps"], e["pe"], e["onset"], thr, w)
        else:
            po = _fp(e["vo"], e["ps"], e["pe"], e["onset"], thr, w)
        out.append(np.nan if po is None else (po - e["onset"]) / FRAME_HZ)
    return np.array(out)


def _fp(sig, ps, pe, onset, thr, w):
    s = moving_average(sig, w); n = len(s); end = min(n, onset + FRAME_HZ)
    c = ps if (ps < n and s[ps] >= thr) else None
    if c is None:
        for i in range(ps + 1, min(pe, n)):
            if s[i] >= thr and s[i - 1] < thr:
                c = i; break
    if c is None:
        return None
    p = c
    for i in range(c + 1, end):
        if s[i] > s[p]:
            p = i
        elif s[i] < s[p]:
            break
    return p + 1


def pnow_errs(events, thr):
    return np.array([(lambda po: np.nan if po is None else (po - e["onset"]) / FRAME_HZ)(
        localize_firstpeak(e["pnow"], e["ps"], e["onset"], thr, 10)) for e in events])


def pf_errs(events, thr, ma, delta):
    out = []
    for e in events:
        c = crossing_in_region(moving_average(e["pfut"], ma), e["ps"], e["pe"], thr)
        out.append(np.nan if c is None else (c + round(delta * FRAME_HZ) - e["onset"]) / FRAME_HZ)
    return np.array(out)


def eo_errs(events, thr, anchor, ma):
    """EO(期待 first active bin)の誤差。predict_event を利用。"""
    out = []
    for e in events:
        po = predict_event(e, "EO", thr, anchor, ma, 0.0)
        out.append(np.nan if po is None else (po - e["onset"]) / FRAME_HZ)
    return np.array(out)


def select_eo(val_ev):
    best = None
    for ma in [1, 5, 10]:
        for anchor in ["cross", "last"]:
            for thr in np.round(np.arange(0.05, 0.96, 0.05), 2):
                _, _, wi, _ = summ(eo_errs(val_ev, thr, anchor, ma))
                if best is None or wi[0.3] > best[0]:
                    best = (wi[0.3], thr, anchor, ma)
    return best[1], best[2], best[3]


def select_vapo(val_ev):
    best = None
    for peak in ["argmax", "first_peak"]:
        for w in [1, 5, 10]:
            for thr in np.round(np.arange(0.15, 0.71, 0.05), 2):
                _, _, wi, _ = summ(vapo_errs(val_ev, thr, w, peak))
                if best is None or wi[0.3] > best[0]:
                    best = (wi[0.3], peak, w, thr)
    return best[1], best[2], best[3]


def select_pnow(val_ev):
    best = (0.3, -1)
    for thr in np.round(np.arange(0.30, 0.71, 0.05), 2):
        _, _, wi, _ = summ(pnow_errs(val_ev, thr))
        det = np.mean(~np.isnan(pnow_errs(val_ev, thr)))
        if wi[0.5] > best[1]:
            best = (thr, wi[0.5])
    return best[0]


def select_pf(val_ev):
    best = None
    for ma in [1, 5, 10]:
        for thr in np.round(np.arange(0.05, 0.96, 0.05), 2):
            for dl in np.round(np.arange(0.0, 2.01, 0.1), 1):
                _, _, wi, _ = summ(pf_errs(val_ev, thr, ma, dl))
                if best is None or wi[0.3] > best[0]:
                    best = (wi[0.3], thr, ma, dl)
    return best[1], best[2], best[3]


def holm(pvals):
    m = len(pvals); order = np.argsort(pvals); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * pvals[i]); adj[i] = min(1.0, run)
    return adj.tolist()


def cohens_h(p1, p2):
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def paired(vo_e, ref_e):
    """VAP-O vs ref の対応検定。出力: dict(±τ McNemar Holm p, h(±0.1), dz, wilcoxon)。"""
    res = {}
    raw = []
    for t in TOLS:
        vh = np.nan_to_num(np.abs(vo_e), nan=1e9) <= t
        rh = np.nan_to_num(np.abs(ref_e), nan=1e9) <= t
        n10 = int(np.sum(vh & ~rh)); n01 = int(np.sum(~vh & rh))
        p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        raw.append(p)
        res[f"vo_{t}"] = float(vh.mean()); res[f"ref_{t}"] = float(rh.mean())
    for t, ap in zip(TOLS, holm(raw)):
        res[f"holm_{t}"] = ap
    res["h_0.1"] = cohens_h(res["vo_0.1"], res["ref_0.1"])
    mask = ~np.isnan(vo_e) & ~np.isnan(ref_e)
    a, b = np.abs(vo_e[mask]), np.abs(ref_e[mask])
    diff = b - a
    res["dz"] = float(diff.mean() / diff.std(ddof=1)) if len(diff) > 1 else np.nan
    res["wilcoxon"] = float(stats.wilcoxon(a, b).pvalue) if mask.sum() > 10 else np.nan
    res["n"] = int(mask.sum())
    return res


def main():
    os.chdir(_ROOT)
    per_seed = {}
    for sd, (vpat, bpat) in SEEDS.items():
        cache = Path(f"reports/_msstats_{sd}.pkl")
        if cache.exists():
            val_ev, test_ev = pickle.load(open(cache, "rb"))
            print(f"[{sd}] loaded cache ({len(test_ev)} test events)")
        else:
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            torch.serialization.add_safe_globals([VapConfig, OptConfig, DataConfig, EventConfig,
                                                  BaselineVAPModel, MaaiVapConfig])
            vapo = VAPModel.load_from_checkpoint(best_ckpt(vpat, "onset"), map_location=dev).to(dev)
            if hasattr(vapo.conf, "text_causal"):
                vapo.conf.text_causal = False
            baseb = BaselineVAPModel.load_from_checkpoint(best_ckpt(bpat, "val"), map_location=dev).to(dev)
            obj = ObjectiveVAP(bin_times=[0.2, 0.4, 0.6, 0.8], frame_hz=FRAME_HZ)

            def loader(csv):
                d = VapDataset(path=csv, window_size=20.0, stride=20.0, cpc_feature_dir=CPC_DIR)
                return DataLoader(d, batch_size=4, num_workers=0, shuffle=False)

            random.seed(SEED_NEG); np.random.seed(SEED_NEG); torch.manual_seed(SEED_NEG)
            print(f"[{sd}] collect val..."); val_ev = collect_both(vapo, baseb, obj, loader(VAL_CSV))
            random.seed(SEED_NEG); np.random.seed(SEED_NEG); torch.manual_seed(SEED_NEG)
            print(f"[{sd}] collect test..."); test_ev = collect_both(vapo, baseb, obj, loader(TEST_CSV))
            pickle.dump((val_ev, test_ev), open(cache, "wb"))

        # val 選択
        vpeak, vw, vthr = select_vapo(val_ev)
        pn_thr = select_pnow(val_ev)
        pf_thr, pf_ma, pf_dl = select_pf(val_ev)
        # test 誤差
        vo_e = vapo_errs(test_ev, vthr, vw, vpeak)
        pn_e = pnow_errs(test_ev, pn_thr)
        G = float(np.median([(e["onset"] - e["pe"]) / FRAME_HZ for e in val_ev]))
        fg_e = np.array([abs(G - (e["onset"] - e["pe"]) / FRAME_HZ) for e in test_ev])
        pf_e = pf_errs(test_ev, pf_thr, pf_ma, pf_dl)
        eo_thr, eo_anchor, eo_ma = select_eo(val_ev)
        eo_e = eo_errs(test_ev, eo_thr, eo_anchor, eo_ma)
        det, mae, within, _ = summ(vo_e)
        pn_s = summ(pn_e); fg_s = summ(fg_e); pf_s = summ(pf_e); eo_s = summ(eo_e)
        per_seed[sd] = dict(
            n=len(test_ev), vo_cfg=f"{vpeak}/MA{vw}/thr{vthr}", G=G,
            pf_cfg=f"thr{pf_thr}/MA{pf_ma}/δ{pf_dl}", eo_cfg=f"thr{eo_thr}/{eo_anchor}/MA{eo_ma}",
            vo_det=det, vo_mae=mae, vo_within=within,
            vs_pnow=paired(vo_e, pn_e), vs_fg=paired(vo_e, fg_e), vs_pf=paired(vo_e, pf_e),
            # 参照の全 τ（Table 2 用）
            pnow=dict(det=pn_s[0], mae=pn_s[1], within=pn_s[2]),
            fg=dict(det=fg_s[0], mae=fg_s[1], within=fg_s[2]),
            pf=dict(det=pf_s[0], mae=pf_s[1], within=pf_s[2]),
            eo=dict(det=eo_s[0], mae=eo_s[1], within=eo_s[2]))
        print(f"[{sd}] done: VAP-O ±0.1={within[0.1]:.3f} MAE={mae:.3f}")

    sk = list(SEEDS.keys())

    def ms(vals):
        a = np.array([v for v in vals if v is not None and not np.isnan(v)], float)
        return f"{a.mean():.3f}±{a.std(ddof=1):.3f}"

    L = ["# 複数 seed での対応統計検定（seed 42/43/44, 沈黙経由 shift）\n",
         "- 各 seed で両モデルを再推論し同一イベント上で対応検定。ハイパラは各 seed の val で選択。",
         "- 参照: p_now(反応的), 固定ギャップ(G=val中央値), p_future+δ(検証調整済み最強読み出し)。",
         "- ±τ 的中は被覆補正込み。McNemar は許容幅ファミリー内 Holm 補正。\n"]

    # Table 2（タイミング, 全予測器 × 全 τ, mean±std）
    def msv(getter):
        return ms([getter(per_seed[s]) for s in sk])

    L.append("## Table 2 用（沈黙経由タイミング, mean±std over seeds）")
    L.append("| 予測器 | det | MAE(s) | ±0.1s | ±0.3s | ±0.5s | ±1.0s |")
    L.append("|---|---|---|---|---|---|---|")
    def prow(name, key):
        if key == "vo":
            g = lambda d, f: d[f"vo_{ 'det' if f=='det' else 'mae' if f=='mae' else 'within'}"] if f in ("det","mae") else d["vo_within"][f]
            det = msv(lambda d: d["vo_det"]); mae = msv(lambda d: d["vo_mae"])
            w = {t: msv(lambda d: d["vo_within"][t]) for t in TOLS}
        else:
            det = msv(lambda d: d[key]["det"]); mae = msv(lambda d: d[key]["mae"])
            w = {t: msv(lambda d: d[key]["within"][t]) for t in TOLS}
        L.append(f"| {name} | {det} | {mae} | {w[0.1]} | {w[0.3]} | {w[0.5]} | {w[1.0]} |")
    prow("VAP-O", "vo")
    prow("固定ギャップ", "fg")
    prow("VAP p_now (反応的)", "pnow")
    prow("VAP EO (期待first bin)", "eo")
    prow("VAP p_future+δ", "pf")
    L.append("")

    # 対応検定（±0.1s 中心）
    L.append("## 対応検定 ±0.1s（VAP-O vs 各参照, 全 seed）")
    L.append("| 参照 | VAP-O ±0.1 (mean±std) | 参照 ±0.1 (mean±std) | Cohen's h (mean±std) | 最保守 Holm p | 全seed有意 |")
    L.append("|---|---|---|---|---|---|")
    for key, name in [("vs_pnow", "p_now(反応的)"), ("vs_fg", "固定ギャップ"), ("vs_pf", "p_future+δ")]:
        vo01 = [per_seed[s][key]["vo_0.1"] for s in sk]
        rf01 = [per_seed[s][key]["ref_0.1"] for s in sk]
        hs = [per_seed[s][key]["h_0.1"] for s in sk]
        maxp = max(per_seed[s][key]["holm_0.1"] for s in sk)
        sig = "はい" if maxp < 0.05 else "いいえ"
        L.append(f"| {name} | {ms(vo01)} | {ms(rf01)} | {ms(hs)} | {maxp:.2e} | {sig} |")
    L.append("")
    L.append("## seed 別詳細（±0.1s McNemar Holm p / Wilcoxon p / dz）")
    for key, name in [("vs_pnow", "p_now"), ("vs_fg", "固定ギャップ"), ("vs_pf", "p_future+δ")]:
        L.append(f"\n### VAP-O vs {name}")
        L.append("| seed | VAP-O ±0.1 | 参照 ±0.1 | Holm p(±0.1) | Wilcoxon p | dz |")
        L.append("|---|---|---|---|---|---|")
        for s in sk:
            r = per_seed[s][key]
            L.append(f"| {s} | {r['vo_0.1']*100:.1f}% | {r['ref_0.1']*100:.1f}% | "
                     f"{r['holm_0.1']:.2e} | {r['wilcoxon']:.2e} | {r['dz']:.2f} |")
    L.append("")
    L.append("選択設定: " + "; ".join(
        f"seed{s}: VAP-O={per_seed[s]['vo_cfg']}, G={per_seed[s]['G']:.2f}, PF={per_seed[s]['pf_cfg']}"
        for s in sk))
    out = Path("reports/multi_seed_stats.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L)); print("\nsaved ->", out)


if __name__ == "__main__":
    main()
