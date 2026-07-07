"""Tier 1b: ベースラインとの対応のある有意差検定（再学習なし・推論のみ）。

役割: VAP-O(stereo-onset) と各ベースラインを **同一バッチ・同一イベント抽出・同一読み出し**
      で評価し、同一イベント上での対応のある検定を行う。
  (1) 検出: VAP-O vs Baseline B（256クラスVAP）。両者をイベント単位に統一
      （Base B も native のフレーム単位でなく VAP-O と同じ last/max 集約）。
      → イベントごとの正誤で McNemar。
  (2) タイミング: VAP-O vs VAP-onset（Base B の p_now を発話開始場所に解釈）。
      → 同一 shift/overlap イベントで ±τ 的中(McNemar) と絶対誤差(Wilcoxon)。

注: 公平性のため両モデルに同じイベント領域・同じ「次話者チャネル」規約を適用する
    （neg は他話者チャネル）。閾値は各モデルとも val で最適化し test に適用。

出力: reports/tier1b_paired.md
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "trains"))
sys.path.insert(0, str(_ROOT / "vap")); sys.path.insert(0, str(_ROOT / "scripts"))

from config import EventConfig, VapConfig, OptConfig
from config.training import DataConfig
from datasets.dataset import VapDataset
from train import VAPModel
from train_baseline import BaselineVAPModel
from objective import ObjectiveVAP
from vap.events import TurnTakingEvents
from model import VapConfig as MaaiVapConfig

FRAME_HZ = 20
TOLS = [0.1, 0.3, 0.5, 1.0]
TEST_CSV = "/Users/onishi/data/switchboard/vap-o_dataset/test.csv"
VAL_CSV = "/Users/onishi/data/switchboard/vap-o_dataset/val.csv"
CPC_DIR = "/Users/onishi/data/switchboard/vap-o_dataset/cpc_features"
VAPO_CKPT = "output/checkpoints_stereo_onset/VapGPT_20Hz_cpc_glove100gru_d256-epoch28-onset_0.28265.ckpt"
BASEB_CKPT = "output/checkpoints_baseline_retrain/baseline_retrain-epoch16-val_2.52317.ckpt"

METRICS = {  # pos/neg キーと Base B の信号（p_now/p_future）
    "hs": {"pos": "shift", "neg": "hold", "neg_flip": True, "bsig": "p_now", "agg": "last"},
    "pred_shift": {"pos": "pred_shift", "neg": "pred_shift_neg", "neg_flip": True, "bsig": "p_future", "agg": "last"},
    "pred_shift_ov": {"pos": "pred_shift_ov", "neg": "pred_shift_ov_neg", "neg_flip": True, "bsig": "p_future", "agg": "max"},
    "bc": {"pos": "pred_backchannel", "neg": "pred_backchannel_neg", "neg_flip": False, "bsig": "p_now", "agg": "max"},
}


def agg(sig, m):
    if len(sig) == 0:
        return 0.0
    return float(sig.max()) if m == "max" else float(sig[-1])


def moving_average(x, w):
    if w <= 1:
        return x
    pad = np.concatenate([np.full(w - 1, x[0]), x])
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def localize_argmax(sig, ps, pe, onset, thr, w=1):
    """VAP-O 用: MA→予測領域内で閾値交差→交差〜onset+1s の大域ピーク(argmax)。"""
    s = moving_average(sig, w)
    n = len(s)
    end = min(n, onset + FRAME_HZ)
    cross = ps if (ps < n and s[ps] >= thr) else None
    if cross is None:
        for i in range(ps + 1, min(pe, n)):
            if s[i] >= thr and s[i - 1] < thr:
                cross = i
                break
    if cross is None:
        return None
    seg_end = max(cross + 1, end)
    return int(cross + np.argmax(s[cross:seg_end])) + 1


def localize_firstpeak(sig, ps, onset, thr, w=10):
    """VAP-onset 用: MA→onset+1s まで閾値交差→交差後の最初の局所ピーク（native）。"""
    s = moving_average(sig, w)
    n = len(s)
    end = min(n, onset + FRAME_HZ)
    cross = ps if s[ps] >= thr else None
    if cross is None:
        for i in range(ps + 1, end):
            if s[i] >= thr and s[i - 1] < thr:
                cross = i
                break
    if cross is None:
        return None
    p = cross
    for i in range(cross + 1, end):
        if s[i] > s[p]:
            p = i
        elif s[i] < s[p]:
            break
    return p + 1


def run_models(vapo, baseb, objective, loader, ee):
    """両モデルを同一バッチで回し、検出用・タイミング用のイベントを集める。
    出力: det[m]=[(vo_score用sig, bb用sig, tgt)], tim[etype]=[dict(vo_sig,bb_sig,ps,pe,onset)]。"""
    det = {m: [] for m in METRICS}
    tim = {"shift": [], "shift_ov": []}
    dev = next(vapo.parameters()).device
    vapo.eval(); baseb.eval()
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
            op = vo["onset_proximity"].cpu().numpy()                      # (B,T,2)
            bo = baseb.forward_from_cpc_features(batch.get("cpc_feat_1"), batch.get("cpc_feat_2"))
            probs = bo["logits"].softmax(dim=-1).cpu()
            p_now = objective.probs_next_speaker_aggregate(probs, 0, 1).numpy()    # (B,T,2)
            p_fut = objective.probs_next_speaker_aggregate(probs, 2, 3).numpy()
            va = batch["va"].cpu()
            ev = ee(va)
            B, T = op.shape[0], op.shape[1]

            # --- 検出（イベント単位, 両モデル同一領域・同一チャネル規約） ---
            for m, c in METRICS.items():
                bsig_full = p_now if c["bsig"] == "p_now" else p_fut
                for key, tgt in [(c["pos"], 1), (c["neg"], 0)]:
                    if key not in ev:
                        continue
                    for b in range(B):
                        for s, e, spk in ev[key][b]:
                            if s >= e or s >= T:
                                continue
                            e2 = min(e, T)
                            nxt = (1 - spk) if (tgt == 0 and c["neg_flip"]) else spk
                            det[m].append((op[b, s:e2, nxt].astype(np.float32),
                                           bsig_full[b, s:e2, nxt].astype(np.float32), tgt))

            # --- タイミング（pred 領域を shift/overlap にマッチ, 次話者=ps_spk） ---
            for etype in ("shift", "shift_ov"):
                pkey = "pred_shift" if etype == "shift" else "pred_shift_ov"
                if etype not in ev or pkey not in ev:
                    continue
                for b in range(B):
                    for ps, pe, ps_spk in ev[pkey][b]:
                        best = None
                        for sil, ons, spk in ev[etype][b]:
                            if spk != ps_spk:
                                continue
                            d = abs(pe - sil)
                            if d <= 5 and (best is None or d < best[0]):
                                best = (d, sil, ons)
                        if best is None:
                            continue
                        _, sil, ons = best
                        true_onset = ons if etype == "shift" else sil
                        if not (0 <= ps < pe <= T and 0 <= true_onset < T):
                            continue
                        tim[etype].append(dict(
                            vo=op[b, :, ps_spk].astype(np.float32),
                            bb=p_now[b, :, ps_spk].astype(np.float32),
                            ps=int(ps), pe=int(pe), onset=int(true_onset)))
    return det, tim


def best_thr_f1(scores, tgts, grid):
    """val で F1 最大の閾値。出力: 閾値。"""
    s, t = np.asarray(scores), np.asarray(tgts)
    bf, bt = -1, grid[0]
    for thr in grid:
        f = f1_score(t, (s >= thr).astype(int), average="weighted", zero_division=0)
        if f > bf:
            bf, bt = f, thr
    return bt


def mcnemar(vo_ok, bb_ok):
    """対応のある正誤配列から McNemar（二項）検定。出力: (n01, n10, p)。"""
    vo_ok, bb_ok = np.asarray(vo_ok), np.asarray(bb_ok)
    n10 = int(np.sum(vo_ok & ~bb_ok))   # VAP-O のみ正
    n01 = int(np.sum(~vo_ok & bb_ok))   # Base のみ正
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    return n01, n10, p


def main():
    os.chdir(_ROOT)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.serialization.add_safe_globals([VapConfig, OptConfig, DataConfig, EventConfig,
                                          BaselineVAPModel, MaaiVapConfig])
    vapo = VAPModel.load_from_checkpoint(VAPO_CKPT, map_location=dev).to(dev)
    if hasattr(vapo.conf, "text_causal"):
        vapo.conf.text_causal = False
    baseb = BaselineVAPModel.load_from_checkpoint(BASEB_CKPT, map_location=dev).to(dev)
    objective = ObjectiveVAP(bin_times=[0.2, 0.4, 0.6, 0.8], frame_hz=FRAME_HZ)
    ee = TurnTakingEvents(EventConfig())

    def loader(csv):
        d = VapDataset(path=csv, window_size=20.0, stride=20.0, cpc_feature_dir=CPC_DIR)
        return DataLoader(d, batch_size=4, num_workers=0, shuffle=False)

    print("collect val...")
    dv, tv = run_models(vapo, baseb, objective, loader(VAL_CSV), ee)
    print("collect test...")
    dt, tt = run_models(vapo, baseb, objective, loader(TEST_CSV), ee)

    L = ["## Tier 1b: ベースラインとの対応のある有意差検定（同一イベント, 推論のみ）\n",
         "VAP-O(stereo-onset) vs Baseline B / VAP-onset。両者を同一イベント・同一読み出し"
         "（イベント単位 last/max, 次話者チャネル）で評価。閾値は val 最適化→test。\n"]

    # === 検出 McNemar ===
    L.append("### 検出 F1（イベント単位, 公平比較）と McNemar")
    L.append("| metric | n | VAP-O F1 | Base B F1 | VAPO正のみ | Base正のみ | p値 | 有意 |")
    L.append("|---|---|---|---|---|---|---|---|")
    grid = np.arange(0.0, 1.0001, 0.01)
    for m, c in METRICS.items():
        a = c["agg"]
        vo_v = [agg(x[0], a) for x in dv[m]]; bb_v = [agg(x[1], a) for x in dv[m]]; yv = [x[2] for x in dv[m]]
        vo_t = np.array([agg(x[0], a) for x in dt[m]]); bb_t = np.array([agg(x[1], a) for x in dt[m]])
        yt = np.array([x[2] for x in dt[m]])
        tv_thr = best_thr_f1(vo_v, yv, grid); tb_thr = best_thr_f1(bb_v, yv, grid)
        vo_pred = (vo_t >= tv_thr).astype(int); bb_pred = (bb_t >= tb_thr).astype(int)
        vo_f1 = f1_score(yt, vo_pred, average="weighted", zero_division=0)
        bb_f1 = f1_score(yt, bb_pred, average="weighted", zero_division=0)
        n01, n10, p = mcnemar(vo_pred == yt, bb_pred == yt)
        sig = "はい" if p < 0.05 else "いいえ"
        L.append(f"| {m} | {len(yt)} | {vo_f1:.3f} | {bb_f1:.3f} | {n10} | {n01} | {p:.2e} | {sig} |")
    L.append("")

    # === タイミング（VAP-O vs VAP-onset） ===
    L.append("### タイミング: VAP-O vs VAP-onset（同一イベント, 対応検定）")
    for etype, label in [("shift", "沈黙経由"), ("shift_ov", "重複経由")]:
        # 閾値選択（val）: VAP-O=within0.3*det, VAP-onset=within0.5*det（各native）
        def vo_err(ev_list, thr):
            e = []
            for d in ev_list:
                po = localize_argmax(d["vo"], d["ps"], d["pe"], d["onset"], thr, 1)
                e.append(None if po is None else abs(po - d["onset"]) / FRAME_HZ)
            return e

        def bb_err(ev_list, thr):
            e = []
            for d in ev_list:
                po = localize_firstpeak(d["bb"], d["ps"], d["onset"], thr, 10)
                e.append(None if po is None else abs(po - d["onset"]) / FRAME_HZ)
            return e

        def score_of(errs, tol_key):
            ae = np.array([x if x is not None else np.nan for x in errs])
            det = np.mean(~np.isnan(ae))
            within = (np.nan_to_num(ae, nan=1e9) <= tol_key).mean()
            return within * det

        best_vo = max(np.round(np.arange(0.15, 0.71, 0.05), 2),
                      key=lambda th: score_of(vo_err(tv[etype], th), 0.3))
        best_bb = max(np.round(np.arange(0.30, 0.71, 0.05), 2),
                      key=lambda th: score_of(bb_err(tv[etype], th), 0.5))
        vo_e = np.array([x if x is not None else np.nan for x in vo_err(tt[etype], best_vo)])
        bb_e = np.array([x if x is not None else np.nan for x in bb_err(tt[etype], best_bb)])
        n = len(vo_e)
        L.append(f"\n**{label}**（n={n}, VAP-O thr{best_vo}/argmax, VAP-onset thr{best_bb}/firstpeak）")
        L.append(f"- 検出率: VAP-O {np.mean(~np.isnan(vo_e))*100:.0f}% / VAP-onset {np.mean(~np.isnan(bb_e))*100:.0f}%")
        L.append(f"- MAE(検出分): VAP-O {np.nanmean(vo_e):.3f}s / VAP-onset {np.nanmean(bb_e):.3f}s")
        L.append("| τ | VAP-O 的中 | VAP-onset 的中 | VAPO○のみ | VAPonset○のみ | p値 | 有意 |")
        L.append("|---|---|---|---|---|---|---|")
        for t in TOLS:
            vo_hit = np.nan_to_num(vo_e, nan=1e9) <= t   # 非検出=誤り
            bb_hit = np.nan_to_num(bb_e, nan=1e9) <= t
            n01, n10, p = mcnemar(vo_hit, bb_hit)
            sig = "はい" if p < 0.05 else "いいえ"
            L.append(f"| ±{t}s | {vo_hit.mean()*100:.1f}% | {bb_hit.mean()*100:.1f}% | {n10} | {n01} | {p:.2e} | {sig} |")
        # Wilcoxon（両者検出のみ）
        mask = ~np.isnan(vo_e) & ~np.isnan(bb_e)
        if mask.sum() > 10:
            w = stats.wilcoxon(vo_e[mask], bb_e[mask])
            L.append(f"- 絶対誤差 Wilcoxon（両者検出 n={int(mask.sum())}）: "
                     f"VAP-O {vo_e[mask].mean():.3f} vs VAP-onset {bb_e[mask].mean():.3f}, p={w.pvalue:.2e}")
    L.append("")

    out = Path("reports/tier1b_paired.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L)); print("\nsaved ->", out)


if __name__ == "__main__":
    main()
