"""VAP(256クラス) から構成できる timing-aware な onset 読み出し群の公平評価。

役割: 査読指摘「p_now 読み替えは VAP のタイミング情報を過小評価している可能性がある。
      より強い VAP-derived onset baseline を、VAP-O と同等の validation-tuned な
      自由度で評価せよ」に応える。

読み出し（すべて 256 クラス分布から導出, ハイパーパラメータは val のみで選択）:
  EO  期待onset     : 最初に active になる future bin の同時分布 P(K=j) を厳密に計算し
                      （クラスごとに先頭 active bin を求め確率を集約）、
                      条件付き期待遅延 E[delay | K<=3] を予測遅延とする。
  HM  hazard中央値  : 同じ P(K=j) の条件付き累積分布が 0.5 を超える最初の bin の中心。
  EB  earliest-bin  : 周辺 bin 確率 q_k が τ を超える最小 bin の中心を予測遅延とする。
  PF  p_future+δ    : p_future の予測領域内交差 + オフセット δ（δ は val で調整）。
  PN  p_now+δ       : 同上を p_now に（p_now にも同等の調整自由度を与える）。

プロトコル（VAP-O と完全に同一）:
  - 同一イベント集合（pred 領域と shift/shift_ov のマッチング |pe−sil|<=5, 同一窓/順序）。
  - 検出制約: 信頼度信号の交差は予測領域 [ps, pe) 内のみ（VAP-O と同じ）。
  - しきい値・アンカー（交差 or 領域末尾）・MA・δ はすべて val の
    被覆込み ±0.3s 率で選択（VAP-O の readout 選択と同一基準）→ test に 1 回適用。
  - 指標: det / MAE(検出分) / ±τ(被覆込み)。最良読み出しは VAP-O と同一イベントで
    exact McNemar(±0.1, Holm) と Wilcoxon による対応検定。

使い方: .venv/bin/python scripts/vap_timing_readouts.py
出力: reports/vap_timing_readouts.md（キャッシュ reports/_vapreadout_cache.pkl）
"""
import os
import pickle
import random
import sys
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
from train_baseline import BaselineVAPModel
from vap.events import TurnTakingEvents
from model import VapConfig as MaaiVapConfig
from tier1b_paired import BASEB_CKPT, moving_average, localize_argmax
from utilities.paths import switchboard_paths

FRAME_HZ = 20
TOLS = [0.1, 0.3, 0.5, 1.0]
SEED = 42
BIN_CENTERS = np.array([0.1, 0.4, 0.9, 1.6])  # bin 境界 [0,.2,.6,1.2,2.0] の中心
_, VAL_CSV, TEST_CSV, CPC_DIR = switchboard_paths()
CACHE = Path("reports/_vapreadout_cache.pkl")
FINAL_CACHE = Path("reports/_final_cache.pkl")


def collect(model, objective, loader, ee):
    """各 shift/shift_ov イベントについて 256 クラス由来の時系列を集める。
    出力: tim[etype] = [dict(q(T,4), pf5(T,5), pnow(T), pfut(T), ps, pe, onset)]。"""
    # クラスごとの「最初の active bin」(0..3, なければ 4) を前計算
    idx = torch.arange(256)
    states = objective.codebook.decode(idx)  # (256, 2, 4)
    first_bin = torch.full((256, 2), 4, dtype=torch.long)
    for c in range(256):
        for s in range(2):
            nz = torch.nonzero(states[c, s])
            if len(nz):
                first_bin[c, s] = int(nz[0])
    onehot_first = torch.zeros(256, 2, 5)
    for c in range(256):
        for s in range(2):
            onehot_first[c, s, first_bin[c, s]] = 1.0

    tim = {"shift": [], "shift_ov": []}
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(dev)
                    if batch[k].is_floating_point():
                        batch[k] = batch[k].float()
            out = model.forward_from_cpc_features(batch.get("cpc_feat_1"), batch.get("cpc_feat_2"))
            probs = out["logits"].softmax(dim=-1).cpu()  # (B,T,256)
            q = torch.einsum("btc,csk->btsk", probs, states.float()).numpy()        # (B,T,2,4)
            pf5 = torch.einsum("btc,csj->btsj", probs, onehot_first).numpy()        # (B,T,2,5)
            pnow = objective.probs_next_speaker_aggregate(probs, 0, 1).numpy()
            pfut = objective.probs_next_speaker_aggregate(probs, 2, 3).numpy()
            va = batch["va"].cpu()
            ev = ee(va)
            B, T = q.shape[0], q.shape[1]
            for etype in ("shift", "shift_ov"):
                pkey = "pred_shift" if etype == "shift" else "pred_shift_ov"
                if etype not in ev or pkey not in ev:
                    continue
                for b in range(B):
                    for ps, pe, spk in ev[pkey][b]:
                        best = None
                        for a, o, sp2 in ev[etype][b]:
                            if sp2 != spk:
                                continue
                            d = abs(pe - a)
                            if d <= 5 and (best is None or d < best[0]):
                                best = (d, a, o)
                        if best is None:
                            continue
                        _, sil, ons = best
                        true_onset = ons if etype == "shift" else sil
                        if not (0 <= ps < pe <= T and 0 <= true_onset < T):
                            continue
                        tim[etype].append(dict(
                            q=q[b, :, spk].astype(np.float16),
                            pf5=pf5[b, :, spk].astype(np.float16),
                            pnow=pnow[b, :, spk].astype(np.float16),
                            pfut=pfut[b, :, spk].astype(np.float16),
                            ps=int(ps), pe=int(pe), onset=int(true_onset)))
    return tim


def crossing_in_region(sig, ps, pe, thr):
    """予測領域 [ps, pe) 内の最初の上方交差フレーム。無ければ None。"""
    n = len(sig)
    if ps < n and sig[ps] >= thr:
        return ps
    for i in range(ps + 1, min(pe, n)):
        if sig[i] >= thr and sig[i - 1] < thr:
            return i
    return None


def predict_event(e, method, thr, anchor, ma, delta):
    """1 イベントの予測 onset フレーム。検出不可なら None。
    入力: e(イベント), method, しきい値, anchor('cross'/'last'), MA窓, δ(秒)。"""
    ps, pe = e["ps"], e["pe"]
    if method in ("EO", "HM"):
        pf5 = e["pf5"].astype(np.float32)
        conf = moving_average(1.0 - pf5[:, 4], ma)
        c = crossing_in_region(conf, ps, pe, thr)
        if c is None:
            return None
        a = c if anchor == "cross" else max(ps, min(pe, len(conf)) - 1 if pe > len(conf) else pe - 1)
        pj = pf5[a, :4]
        tot = pj.sum()
        if tot <= 1e-6:
            return None
        if method == "EO":
            delay = float((pj / tot * BIN_CENTERS).sum())
        else:
            cdf = np.cumsum(pj / tot)
            k = int(np.searchsorted(cdf, 0.5))
            delay = float(BIN_CENTERS[min(k, 3)])
        return a + int(round(delay * FRAME_HZ))
    if method == "EB":
        qq = e["q"].astype(np.float32)
        conf = moving_average(qq.max(axis=1), ma)
        c = crossing_in_region(conf, ps, pe, thr)
        if c is None:
            return None
        a = c if anchor == "cross" else pe - 1
        above = np.where(qq[a] >= thr)[0]
        if len(above) == 0:
            above = [int(qq[a].argmax())]
        return a + int(round(BIN_CENTERS[int(above[0])] * FRAME_HZ))
    # PF / PN: 交差 + δ
    sig = e["pfut" if method == "PF" else "pnow"].astype(np.float32)
    c = crossing_in_region(moving_average(sig, ma), ps, pe, thr)
    if c is None:
        return None
    return c + int(round(delta * FRAME_HZ))


def eval_config(events, method, thr, anchor, ma, delta):
    """設定 1 つの指標。出力: dict(det, MAE, within{τ})。within は被覆込み。"""
    errs = []
    n = len(events)
    for e in events:
        po = predict_event(e, method, thr, anchor, ma, delta)
        errs.append(np.nan if po is None else (po - e["onset"]) / FRAME_HZ)
    ae = np.abs(np.array(errs))
    det = float(np.mean(~np.isnan(ae)))
    mae = float(np.nanmean(ae)) if det > 0 else np.nan
    within = {t: float((np.nan_to_num(ae, nan=1e9) <= t).mean()) for t in TOLS}
    return dict(det=det, MAE=mae, within=within, errs=np.array(errs))


def sweep_method(val_ev, test_ev, method):
    """method のハイパーパラメータを val（被覆込み ±0.3s）で選び test を評価。"""
    thrs = np.round(np.arange(0.05, 0.96, 0.05), 2)
    anchors = ["cross", "last"] if method in ("EO", "HM", "EB") else ["cross"]
    deltas = np.round(np.arange(0.0, 2.01, 0.1), 1) if method in ("PF", "PN") else [0.0]
    best = None
    for ma in [1, 5, 10]:
        for anchor in anchors:
            for thr in thrs:
                for dl in deltas:
                    m = eval_config(val_ev, method, thr, anchor, ma, dl)
                    sc = m["within"][0.3]
                    if best is None or sc > best[0]:
                        best = (sc, thr, anchor, ma, dl)
    _, thr, anchor, ma, dl = best
    mt = eval_config(test_ev, method, thr, anchor, ma, dl)
    tag = f"thr{thr}/MA{ma}" + (f"/{anchor}" if method in ("EO", "HM", "EB") else "") + \
          (f"/δ{dl}" if method in ("PF", "PN") else "")
    return tag, mt


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj, run = np.empty(m), 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * pvals[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def main():
    os.chdir(_ROOT)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    if CACHE.exists():
        d = pickle.load(open(CACHE, "rb"))
        tv, tt = d["val"], d["test"]
        print("loaded cache")
    else:
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        torch.serialization.add_safe_globals([VapConfig, OptConfig, DataConfig, EventConfig,
                                              BaselineVAPModel, MaaiVapConfig])
        model = BaselineVAPModel.load_from_checkpoint(BASEB_CKPT, map_location=dev).to(dev)
        objective = ObjectiveVAP(bin_times=[0.2, 0.4, 0.6, 0.8], frame_hz=FRAME_HZ)

        def loader(csv):
            ds = VapDataset(path=csv, window_size=20.0, stride=20.0, cpc_feature_dir=CPC_DIR)
            return DataLoader(ds, batch_size=4, num_workers=0, shuffle=False)

        print("collect val...")
        tv = collect(model, objective, loader(VAL_CSV), TurnTakingEvents(EventConfig()))
        print("collect test...")
        tt = collect(model, objective, loader(TEST_CSV), TurnTakingEvents(EventConfig()))
        pickle.dump({"val": tv, "test": tt, "meta": {"ckpt": BASEB_CKPT, "seed": SEED}},
                    open(CACHE, "wb"))

    # VAP-O 側の per-event 誤差（final_stats と同じ採用読み出し argmax/MA1/thr0.15）
    fc = pickle.load(open(FINAL_CACHE, "rb"))["data"]
    _, ftv, _, ftt = fc
    vo_err = {}
    for et in ("shift", "shift_ov"):
        assert len(ftt[et]) == len(tt[et]), f"event count mismatch {et}"
        for a, b in zip(ftt[et], tt[et]):
            assert (a["ps"], a["pe"], a["onset"]) == (b["ps"], b["pe"], b["onset"]), "event align"
        errs = []
        for e in ftt[et]:
            po = localize_argmax(e["vo"], e["ps"], e["pe"], e["onset"], 0.15, 1)
            errs.append(np.nan if po is None else (po - e["onset"]) / FRAME_HZ)
        vo_err[et] = np.array(errs)

    L = ["# VAP(256クラス) 由来の timing-aware 読み出し群の評価（査読対応）\n",
         "- 全読み出しは 256 クラス分布から導出。ハイパーパラメータ（しきい値・アンカー・MA・δ）は",
         "  **val の被覆込み ±0.3s 率で選択**（VAP-O の readout 選択と同一基準）し test に 1 回適用。",
         "- 検出制約は VAP-O と同一: 信頼度の交差は予測領域内のみ。イベント集合・窓・順序も同一",
         "  （VAP-O 側キャッシュと ps/pe/onset の一致を assert 済み）。±τ は被覆込み。",
         "- EO=期待onset（最初 active bin の同時分布の条件付き期待値）, HM=同分布の中央値 bin,",
         "  EB=earliest bin 交差, PF/PN=p_future/p_now 交差+δ（δ も val 調整）。\n"]

    best_by_et = {}
    for et, name in [("shift", "沈黙経由"), ("shift_ov", "重複経由")]:
        L.append(f"## {name}（n={len(tt[et])}）")
        L.append("| 読み出し | 採用設定(val) | det | MAE(s) | ±0.1 | ±0.3 | ±0.5 | ±1.0 |")
        L.append("|---|---|---|---|---|---|---|---|")
        # VAP-O 参照行
        ae = np.abs(vo_err[et])
        det = float(np.mean(~np.isnan(ae)))
        w = {t: float((np.nan_to_num(ae, nan=1e9) <= t).mean()) for t in TOLS}
        L.append(f"| **VAP-O (参照)** | argmax/MA1/thr0.15 | {det*100:.0f}% | {np.nanmean(ae):.3f} | "
                 f"{w[0.1]*100:.1f}% | {w[0.3]*100:.1f}% | {w[0.5]*100:.1f}% | {w[1.0]*100:.1f}% |")
        best = None
        for method in ["EO", "HM", "EB", "PF", "PN"]:
            tag, m = sweep_method(tv[et], tt[et], method)
            wi = m["within"]
            L.append(f"| VAP {method} | {tag} | {m['det']*100:.0f}% | {m['MAE']:.3f} | "
                     f"{wi[0.1]*100:.1f}% | {wi[0.3]*100:.1f}% | {wi[0.5]*100:.1f}% | {wi[1.0]*100:.1f}% |")
            if best is None or wi[0.3] > best[2]["within"][0.3]:
                best = (method, tag, m)
        best_by_et[et] = best
        L.append("")

    # 最良 VAP 読み出し vs VAP-O の対応検定
    L.append("## 対応検定: VAP-O vs 最良の VAP-derived 読み出し（同一イベント）")
    for et, name in [("shift", "沈黙経由"), ("shift_ov", "重複経由")]:
        method, tag, m = best_by_et[et]
        b_err = m["errs"]
        v_err = vo_err[et]
        raw, rows = [], []
        for t in TOLS:
            vh = np.nan_to_num(np.abs(v_err), nan=1e9) <= t
            bh = np.nan_to_num(np.abs(b_err), nan=1e9) <= t
            n10 = int(np.sum(vh & ~bh)); n01 = int(np.sum(~vh & bh))
            p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            raw.append(p); rows.append((t, vh.mean(), bh.mean(), n10, n01, p))
        adj = holm(raw)
        L.append(f"\n### {name}: VAP-O vs VAP {method}（{tag}）")
        L.append("| τ | VAP-O | VAP最良 | VAPO○のみ | VAP○のみ | p (exact McNemar) | Holm p | 有意 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for (t, v, b, n10, n01, p), ap in zip(rows, adj):
            L.append(f"| ±{t}s | {v*100:.1f}% | {b*100:.1f}% | {n10} | {n01} | {p:.2e} | {ap:.2e} | "
                     f"{'はい' if ap < 0.05 else 'いいえ'} |")
        mask = ~np.isnan(v_err) & ~np.isnan(b_err)
        if mask.sum() > 10:
            va_, ba_ = np.abs(v_err[mask]), np.abs(b_err[mask])
            wres = stats.wilcoxon(va_, ba_)
            diff = ba_ - va_
            dz = float(diff.mean() / diff.std(ddof=1))
            L.append(f"- 絶対誤差（両者検出 n={int(mask.sum())}）: VAP-O {va_.mean():.3f} vs "
                     f"VAP {method} {ba_.mean():.3f}, Wilcoxon p={wres.pvalue:.2e}, dz={dz:.2f}")

    out = Path("reports/vap_timing_readouts.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L)); print("\nsaved ->", out)


if __name__ == "__main__":
    main()
