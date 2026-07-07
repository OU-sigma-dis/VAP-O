"""論文用の正準統計値を一括算出する最終スクリプト（監査指摘を全て反映）。

役割: これまでの tier1 系スクリプトを置き換える単一の情報源（single source of truth）。
  監査で見つかった問題への対応:
  - [リーク] 集約方式・MA窓・readout・しきい値の選択をすべて val のみで実施。
  - [再現性] random.seed / np seed を固定し、負例サンプリングを再現可能に。
    split ごとに TurnTakingEvents を作り直し状態持ち越しを排除。
  - [公平性] 両モデルを同一バッチ・同一イベント集合で評価（VAP-O=イベント単位の
    我々の読み出し, Baseline=原著のフレーム単位判定。Inoue の評価法は改変しない）。
  - [統計] McNemar は exact binomial。多重比較は τ ファミリー内 Holm 補正。
    効果量は dz（対応, 主）, d（プールSD, 参考）, h（±0.1 被覆込み比率）,
    符号付き rank-biserial。固定ギャップ G は val の中央値（test を見ない）。
  - [表記] ±τ% は被覆補正込み（非検出=誤り）に統一。MAE は検出分のみ＋det% 併記。

使い方: .venv/bin/python scripts/final_stats.py   （キャッシュ reports/_final_cache.pkl 利用）
出力: reports/final_stats.md
"""
import os
import pickle
import random
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
from objective import ObjectiveVAP
from train import VAPModel
from train_baseline import BaselineVAPModel
from vap.events import TurnTakingEvents
from model import VapConfig as MaaiVapConfig
from tier1b_paired import run_models, localize_argmax, localize_firstpeak, VAPO_CKPT, BASEB_CKPT

FRAME_HZ = 20
TOLS = [0.1, 0.3, 0.5, 1.0]
N_BOOT = 2000
SEED = 42
TEST_CSV = "/Users/onishi/data/switchboard/vap-o_dataset/test.csv"
VAL_CSV = "/Users/onishi/data/switchboard/vap-o_dataset/val.csv"
CPC_DIR = "/Users/onishi/data/switchboard/vap-o_dataset/cpc_features"
CACHE = Path("reports/_final_cache.pkl")
METRIC_KEYS = ["hs", "pred_shift", "pred_shift_ov", "bc"]
AGG_CANDS = ["max", "last", "mean"]  # VAP-O のイベント集約候補（val で選択）


# ---------- 基本ユーティリティ ----------

def agg(sig, m):
    if len(sig) == 0:
        return 0.0
    if m == "max":
        return float(sig.max())
    if m == "last":
        return float(sig[-1])
    return float(sig.mean())


def wf1(tgts, preds):
    return f1_score(tgts, preds, average="weighted", zero_division=0)


def best_thr(scores, tgts, grid):
    s, t = np.asarray(scores), np.asarray(tgts)
    bf, bt = -1.0, grid[0]
    for thr in grid:
        f = wf1(t, (s >= thr).astype(int))
        if f > bf:
            bf, bt = f, thr
    return bt, bf


def holm(pvals):
    """Holm 補正。入力: p値リスト。出力: 補正後p値リスト（同順）。"""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def cohens_h(p1, p2):
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def eff_label(x):
    a = abs(x)
    return "大" if a >= 0.8 else ("中" if a >= 0.5 else ("小" if a >= 0.2 else "微小"))


def mcnemar_exact(a_ok, b_ok):
    """exact binomial McNemar。出力: (aのみ正, bのみ正, p)。"""
    a_ok, b_ok = np.asarray(a_ok), np.asarray(b_ok)
    n10 = int(np.sum(a_ok & ~b_ok))
    n01 = int(np.sum(~a_ok & b_ok))
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    return n10, n01, p


def rank_biserial(diff):
    """符号付き rank-biserial（Wilcoxon 効果量）。diff>0 が「提案が良い」。"""
    d = diff[diff != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    r_pos = r[d > 0].sum()
    r_neg = r[d < 0].sum()
    return float((r_pos - r_neg) / (r_pos + r_neg))


def paired_stats(err_a, err_b):
    """対応のある絶対誤差[s]の検定＋効果量。b - a > 0 で a(提案) が良い。
    両者検出のイベントのみ（サブセットである旨は表の脚注で開示）。"""
    mask = ~np.isnan(err_a) & ~np.isnan(err_b)
    a, b = err_a[mask], err_b[mask]
    diff = b - a
    t = stats.ttest_rel(a, b)
    w = stats.wilcoxon(a, b)
    dz = float(diff.mean() / diff.std(ddof=1))
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d = float(diff.mean() / pooled)
    return dict(n=int(mask.sum()), a_mae=float(a.mean()), b_mae=float(b.mean()),
                dz=dz, d=d, r_rb=rank_biserial(diff),
                t_p=float(t.pvalue), w_p=float(w.pvalue))


# ---------- タイミング読み出し ----------

def vo_errors(events, thr, w, peak):
    """VAP-O の per-event 絶対誤差（NaN=非検出）。"""
    out = []
    for d in events:
        if peak == "argmax":
            po = localize_argmax(d["vo"], d["ps"], d["pe"], d["onset"], thr, w)
        else:
            po = _first_peak_vo(d["vo"], d["ps"], d["pe"], d["onset"], thr, w)
        out.append(np.nan if po is None else abs(po - d["onset"]) / FRAME_HZ)
    return np.array(out)


def _first_peak_vo(sig, ps, pe, onset, thr, w):
    """VAP-O 用 first_peak 変種（交差は pred 領域内, argmax との val 比較用）。"""
    from tier1b_paired import moving_average
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
    p = cross
    for i in range(cross + 1, end):
        if s[i] > s[p]:
            p = i
        elif s[i] < s[p]:
            break
    return p + 1


def onset_errors(events, thr):
    """VAP-onset（原著 readout: MA10, first_peak, 広い交差窓）の絶対誤差。"""
    out = []
    for d in events:
        po = localize_firstpeak(d["bb"], d["ps"], d["onset"], thr, 10)
        out.append(np.nan if po is None else abs(po - d["onset"]) / FRAME_HZ)
    return np.array(out)


def timing_summary(ae):
    """絶対誤差配列から det / MAE(検出分) / ±τ(被覆込み) を算出。"""
    det = float(np.mean(~np.isnan(ae)))
    mae = float(np.nanmean(ae)) if det > 0 else np.nan
    within = {t: float((np.nan_to_num(ae, nan=1e9) <= t).mean()) for t in TOLS}
    return det, mae, within


def timing_ci(ae, rng):
    """イベント単位ブートストラップCI。出力: (MAE_ci, within_ci)。"""
    n = len(ae)
    b_mae, b_w = [], {t: [] for t in TOLS}
    for _ in range(N_BOOT):
        s = ae[rng.integers(0, n, n)]
        d = s[~np.isnan(s)]
        if len(d):
            b_mae.append(d.mean())
        for t in TOLS:
            b_w[t].append((np.nan_to_num(s, nan=1e9) <= t).mean())
    pct = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return pct(b_mae), {t: pct(b_w[t]) for t in TOLS}


# ---------- メイン ----------

def main():
    os.chdir(_ROOT)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if CACHE.exists():
        dv, tv, dt, tt = pickle.load(open(CACHE, "rb"))["data"]
        print("loaded cache:", CACHE)
    else:
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        torch.serialization.add_safe_globals([VapConfig, OptConfig, DataConfig, EventConfig,
                                              BaselineVAPModel, MaaiVapConfig])
        vapo = VAPModel.load_from_checkpoint(VAPO_CKPT, map_location=dev).to(dev)
        if hasattr(vapo.conf, "text_causal"):
            vapo.conf.text_causal = False
        baseb = BaselineVAPModel.load_from_checkpoint(BASEB_CKPT, map_location=dev).to(dev)
        objective = ObjectiveVAP(bin_times=[0.2, 0.4, 0.6, 0.8], frame_hz=FRAME_HZ)

        def loader(csv):
            d = VapDataset(path=csv, window_size=20.0, stride=20.0, cpc_feature_dir=CPC_DIR)
            return DataLoader(d, batch_size=4, num_workers=0, shuffle=False)

        # split ごとに extractor を作り直し状態持ち越しを排除
        print("collect val...")
        dv, tv = run_models(vapo, baseb, objective, loader(VAL_CSV), TurnTakingEvents(EventConfig()))
        print("collect test...")
        dt, tt = run_models(vapo, baseb, objective, loader(TEST_CSV), TurnTakingEvents(EventConfig()))
        pickle.dump({"data": (dv, tv, dt, tt),
                     "meta": {"vapo": VAPO_CKPT, "baseb": BASEB_CKPT, "seed": SEED}},
                    open(CACHE, "wb"))

    rng = np.random.default_rng(SEED)
    L = ["# 論文用 正準統計値（final_stats, 監査反映版）\n",
         f"- モデル: VAP-O={Path(VAPO_CKPT).name} / Baseline={Path(BASEB_CKPT).name}",
         f"- seed={SEED}, ブートストラップ{N_BOOT}回, 同一バッチ・同一イベント集合で両モデルを評価",
         "- ハイパーパラメータ（集約方式・MA窓・readout・しきい値）は **val のみ**で選択し test に1回適用",
         "- ±τ% は被覆補正込み（非検出=誤り）。MAE は検出イベントのみ（det% 併記）。\n"]

    # ===== 1. 検出 F1 =====
    L.append("## 1. 検出 F1（VAP-O=イベント単位読み出し / Baseline=原著フレーム単位判定）")
    L.append("| metric | n_event | VAP-O 集約(val選択) | VAP-O F1 [95%CI] | Baseline F1 [95%CI] | 判定 |")
    L.append("|---|---|---|---|---|---|")
    det_summary = {}
    for m in METRIC_KEYS:
        grid_v = np.arange(0.0, 1.0001, 0.005)
        # VAP-O: val で (集約, しきい値) を選択
        best_a, best_f, best_t = None, -1.0, None
        for a in AGG_CANDS:
            vs = [agg(x[0], a) for x in dv[m]]
            vt = [x[2] for x in dv[m]]
            t_, f_ = best_thr(vs, vt, grid_v)
            if f_ > best_f:
                best_a, best_f, best_t = a, f_, t_
        ts = np.array([agg(x[0], best_a) for x in dt[m]])
        yt = np.array([x[2] for x in dt[m]])
        vo_pred = (ts >= best_t).astype(int)
        vo_f1 = wf1(yt, vo_pred)
        # VAP-O CI（イベント単位）
        boots = np.empty(N_BOOT)
        n = len(yt)
        for i in range(N_BOOT):
            idx = rng.integers(0, n, n)
            boots[i] = wf1(yt[idx], vo_pred[idx])
        vo_lo, vo_hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)

        # Baseline: 原著（フレーム単位, グリッド 0.01..0.99）。同一イベントの bb_sig を使用。
        grid_b = np.arange(0.01, 0.991, 0.01)
        fv = np.concatenate([x[1] for x in dv[m]])
        fvt = np.concatenate([np.full(len(x[1]), x[2]) for x in dv[m]])
        bt_, _ = best_thr(fv, fvt, grid_b)
        ft = [x[1] for x in dt[m]]
        ftt = [np.full(len(x[1]), x[2]) for x in dt[m]]
        bb_f1 = wf1(np.concatenate(ftt), (np.concatenate(ft) >= bt_).astype(int))
        # Baseline CI（イベント・ブロック・ブートストラップ）
        bboots = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = rng.integers(0, n, n)
            sc = np.concatenate([ft[j] for j in idx])
            tg = np.concatenate([ftt[j] for j in idx])
            bboots[i] = wf1(tg, (sc >= bt_).astype(int))
        bb_lo, bb_hi = np.percentile(bboots, 2.5), np.percentile(bboots, 97.5)

        overlap = not (vo_lo > bb_hi or bb_lo > vo_hi)
        judge = "重なる（同等）" if overlap else ("VAP-O 上" if vo_lo > bb_hi else "Baseline 上")
        L.append(f"| {m} | {n} | {best_a} | {vo_f1:.3f} [{vo_lo:.3f}, {vo_hi:.3f}] | "
                 f"{bb_f1:.3f} [{bb_lo:.3f}, {bb_hi:.3f}] | {judge} |")
        det_summary[m] = (vo_f1, bb_f1)
    L.append("")

    # ===== 2. タイミング: VAP-O 本体（val で readout/MA/thr 選択） =====
    L.append("## 2. onset タイミング（VAP-O, readout も val で選択）")
    L.append("| event | 選択方式 | n | det | MAE [95%CI] | ±0.1 [CI] | ±0.3 [CI] | ±0.5 [CI] | ±1.0 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    vo_test_err = {}
    vo_params = {}
    for et in ["shift", "shift_ov"]:
        best = None  # (val_sc, peak, w, thr)
        for peak in ["argmax", "first_peak"]:
            for w in [1, 5, 10]:
                for thr in np.round(np.arange(0.15, 0.71, 0.05), 2):
                    ae = vo_errors(tv[et], thr, w, peak)
                    det, _, within = timing_summary(ae)
                    sc = within[0.3]  # 被覆込み ±0.3s（det は within に織込済）
                    if best is None or sc > best[0]:
                        best = (sc, peak, w, thr)
        _, peak, w, thr = best
        ae = vo_errors(tt[et], thr, w, peak)
        vo_test_err[et] = ae
        vo_params[et] = (peak, w, thr)
        det, mae, within = timing_summary(ae)
        mae_ci, w_ci = timing_ci(ae, rng)
        fmt = lambda t: f"{within[t]*100:.1f}% [{w_ci[t][0]*100:.0f},{w_ci[t][1]*100:.0f}]"
        L.append(f"| {et} | {peak}/MA{w}/thr{thr} | {len(ae)} | {det*100:.0f}% | "
                 f"{mae:.3f} [{mae_ci[0]:.3f},{mae_ci[1]:.3f}] | {fmt(0.1)} | {fmt(0.3)} | "
                 f"{fmt(0.5)} | {within[1.0]*100:.1f}% |")
    L.append("")

    # ===== 3. 対応比較: vs VAP-onset =====
    L.append("## 3. 対応比較: VAP-O vs VAP-onset（原著 readout, 同一イベント）")
    for et, name in [("shift", "沈黙経由"), ("shift_ov", "重複経由")]:
        # VAP-onset: 原著プロトコル（val で thr を ±0.5×det で選択）
        bthr, bsc = 0.3, -1
        for thr in np.round(np.arange(0.30, 0.71, 0.05), 2):
            ae = onset_errors(tv[et], thr)
            det, _, within = timing_summary(ae)
            sc = within[0.5]
            if sc > bsc:
                bsc, bthr = sc, thr
        on_ae = onset_errors(tt[et], bthr)
        vo_ae = vo_test_err[et]
        on_det, on_mae, on_w = timing_summary(on_ae)
        vo_det, vo_mae, vo_w = timing_summary(vo_ae)
        ps = paired_stats(vo_ae, on_ae)
        L.append(f"\n### {name}（n={len(vo_ae)}, VAP-onset thr{bthr}）")
        L.append(f"- det: VAP-O {vo_det*100:.0f}% / VAP-onset {on_det*100:.0f}%; "
                 f"MAE(検出分): {vo_mae:.3f} / {on_mae:.3f}")
        # McNemar ±τ（Holm 補正）
        raw_p, rows = [], []
        for t in TOLS:
            vo_hit = np.nan_to_num(vo_ae, nan=1e9) <= t
            on_hit = np.nan_to_num(on_ae, nan=1e9) <= t
            n10, n01, p = mcnemar_exact(vo_hit, on_hit)
            raw_p.append(p)
            rows.append((t, vo_hit.mean(), on_hit.mean(), n10, n01, p))
        adj = holm(raw_p)
        L.append("| τ | VAP-O | VAP-onset | VAPO○のみ | onset○のみ | p (exact McNemar) | Holm補正p | 有意 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for (t, v, o, n10, n01, p), ap in zip(rows, adj):
            sig = "はい" if ap < 0.05 else "いいえ"
            L.append(f"| ±{t}s | {v*100:.1f}% | {o*100:.1f}% | {n10} | {n01} | {p:.2e} | {ap:.2e} | {sig} |")
        L.append(f"- 絶対誤差（両者検出 n={ps['n']}）: t検定 p={ps['t_p']:.2e}, Wilcoxon p={ps['w_p']:.2e}, "
                 f"**dz={ps['dz']:.2f}({eff_label(ps['dz'])})**, d={ps['d']:.2f}, "
                 f"rank-biserial r={ps['r_rb']:.2f}")
        h = cohens_h(vo_w[0.1], on_w[0.1])
        L.append(f"- ±0.1s 的中率（被覆込み）の効果量 Cohen's h = {h:.2f}（{eff_label(h)}）")
    L.append("")

    # ===== 4. 対応比較: vs 固定ギャップ（沈黙経由のみ, G は val 中央値） =====
    gaps_val = np.array([(e["onset"] - e["pe"]) / FRAME_HZ for e in tv["shift"]])
    G = float(np.median(gaps_val))
    gaps_test = np.array([(e["onset"] - e["pe"]) / FRAME_HZ for e in tt["shift"]])
    fg_ae = np.abs(G - gaps_test)
    vo_ae = vo_test_err["shift"]
    ps = paired_stats(vo_ae, fg_ae)
    _, _, vo_w = timing_summary(vo_ae)
    L.append(f"## 4. 対応比較: VAP-O vs 固定ギャップ（G=val中央値 {G:.2f}s, 沈黙経由 n={len(vo_ae)}）")
    raw_p, rows = [], []
    for t in TOLS:
        vo_hit = np.nan_to_num(vo_ae, nan=1e9) <= t
        fg_hit = fg_ae <= t
        n10, n01, p = mcnemar_exact(vo_hit, fg_hit)
        raw_p.append(p)
        rows.append((t, vo_hit.mean(), fg_hit.mean(), n10, n01, p))
    adj = holm(raw_p)
    L.append("| τ | VAP-O | 固定ギャップ | VAPO○のみ | 固定○のみ | p (exact McNemar) | Holm補正p | 有意 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for (t, v, o, n10, n01, p), ap in zip(rows, adj):
        sig = "はい" if ap < 0.05 else "いいえ"
        L.append(f"| ±{t}s | {v*100:.1f}% | {o*100:.1f}% | {n10} | {n01} | {p:.2e} | {ap:.2e} | {sig} |")
    L.append(f"- 絶対誤差（VAP-O検出 n={ps['n']}）: t検定 p={ps['t_p']:.2e}, Wilcoxon p={ps['w_p']:.2e}, "
             f"dz={ps['dz']:.2f}({eff_label(ps['dz'])}), d={ps['d']:.2f}, r={ps['r_rb']:.2f}")
    h = cohens_h(vo_w[0.1], float((fg_ae <= 0.1).mean()))
    L.append(f"- ±0.1s 的中率の効果量 Cohen's h = {h:.2f}（{eff_label(h)}）")
    L.append("")
    L.append("## 開示事項（論文の評価プロトコル節に記載すべき点）")
    L.append("- onset 局在化の探索窓は真の onset+1s でアンカーしたオフライン評価（全手法同一条件）。")
    L.append("- 交差探索窓は VAP-O=予測領域内, VAP-onset=onset+1s まで（原著準拠, ベースライン有利側）。")
    L.append("- F1 は先行研究に従い weighted F1。負例サンプリングは seed=42 で固定。")
    L.append("- dz/d/Wilcoxon は両者検出サブセット、±τ% と h は被覆補正込み全イベント。")

    out = Path("reports/final_stats.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print("\nsaved ->", out)


if __name__ == "__main__":
    main()
