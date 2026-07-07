"""最良評価方式を採用した official 最終値の確定（キャッシュ利用, 再推論なし）。

採用方式:
  検出: hs / pred_shift = 領域「最終フレーム値(last)」, pred_shift_ov / bc = max。
        しきい値は val で F1 最大（細グリッド [0,1] step 0.005）→ test F1。
  タイミング: readout=argmax（大域ピーク=onset proximity の設計上の onset）を基本に、
             MA∈{1,5,10} と first_peak も含めて val(±0.5s内率×被覆)で選び test を評価。
検証: 集約=max が eval①（現行）を再現するかを併記（グリッド妥当性チェック）。
出力: reports/eval_final.md
"""
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

FRAME_HZ = 20
TOLS = [0.1, 0.3, 0.5, 1.0]
CACHE = Path("reports/_sweep_cache.pkl")
METRICS = ["hs", "pred_shift", "pred_shift_ov", "bc"]
# 採用集約（sweep の結果に基づく）
AGG = {"hs": "last", "pred_shift": "last", "pred_shift_ov": "max", "bc": "max"}


def agg(sig, m):
    if len(sig) == 0:
        return 0.0
    if m == "max":
        return float(sig.max())
    if m == "last":
        return float(sig[-1])
    if m == "mean":
        return float(sig.mean())
    raise ValueError(m)


def f1_at_best_thr(val_scores, val_tgt, test_scores, test_tgt):
    grid = np.arange(0.0, 1.0001, 0.005)
    bf, bt = -1, 0.5
    vs, vt = np.asarray(val_scores), np.asarray(val_tgt)
    for t in grid:
        f = f1_score(vt, (vs >= t).astype(int), average="weighted", zero_division=0)
        if f > bf:
            bf, bt = f, t
    ts, tt = np.asarray(test_scores), np.asarray(test_tgt)
    return f1_score(tt, (ts >= bt).astype(int), average="weighted", zero_division=0), bt


def ma(x, w):
    if w <= 1:
        return x
    pad = np.concatenate([np.full(w - 1, x[0]), x])
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def localize(sig, ps, pe, onset, thr, w, peak):
    s = ma(sig, w)
    n = len(s)
    end = min(n, onset + FRAME_HZ)
    cross = None
    if ps < n and s[ps] >= thr:
        cross = ps
    else:
        for i in range(ps + 1, min(pe, n)):  # 交差は予測領域内
            if s[i] >= thr and s[i - 1] < thr:
                cross = i
                break
    if cross is None:
        return None
    if peak == "argmax":
        seg_end = max(cross + 1, end)
        return int(cross + np.argmax(s[cross:seg_end])) + 1
    p = cross  # first_peak
    for i in range(cross + 1, end):
        if s[i] > s[p]:
            p = i
        elif s[i] < s[p]:
            break
    return p + 1


def timing_metrics(events, thr, w, peak):
    errs = []
    for e in events:
        po = localize(e["sig"], e["ps"], e["pe"], e["onset"], thr, w, peak)
        if po is not None:
            errs.append((po - e["onset"]) / FRAME_HZ)
    n = len(events)
    if not errs:
        return dict(n=n, det=0, MAE=np.nan, within={t: 0 for t in TOLS})
    ae = np.abs(np.array(errs))
    return dict(n=n, det=len(errs) / n, MAE=float(ae.mean()),
                within={t: float((ae <= t).mean()) for t in TOLS})


def best_timing(val_ev, test_ev):
    # readout は argmax に固定（onset proximity は設計上 onset でピーク＝原理的に正しい）。
    # (w, thr) の選択は val スコアのみで行う（test を見ない: リーク防止）。
    peak = "argmax"
    best = None  # (val_sc, tag, test_metrics)
    for w in [1, 5, 10]:
        bthr, bsc = 0.3, -1
        for thr in np.round(np.arange(0.15, 0.71, 0.05), 2):
            mv = timing_metrics(val_ev, thr, w, peak)
            sc = mv["within"][0.3] * mv["det"]  # 精密性寄りの基準（±0.3s×被覆）
            if sc > bsc:
                bsc, bthr = sc, thr
        if best is None or bsc > best[0]:
            mt = timing_metrics(test_ev, bthr, w, peak)
            best = (bsc, f"MA{w}/{peak}/thr{bthr}", mt)
    return best[1], best[2]


def main():
    det_val, tim_val, det_test, tim_test = pickle.load(open(CACHE, "rb"))
    L = ["## 最良方式を採用した最終評価（stereo-onset, 音声のみ, 再推論なし）\n"]

    # 検出
    L.append("### 検出 F1（採用集約 vs max=現行, 細グリッド閾値）")
    L.append("| metric | 採用集約 | F1(採用) | F1(max=現行) | Baseline B |")
    L.append("|---|---|---|---|---|")
    baseB = {"hs": 0.788, "pred_shift": 0.705, "pred_shift_ov": 0.746, "bc": 0.831}
    for m in METRICS:
        vv, vt = zip(*det_val[m]); tv, tt = zip(*det_test[m])
        f_adopt, _ = f1_at_best_thr([agg(s, AGG[m]) for s in vv], vt,
                                    [agg(s, AGG[m]) for s in tv], tt)
        f_max, _ = f1_at_best_thr([agg(s, "max") for s in vv], vt,
                                  [agg(s, "max") for s in tv], tt)
        L.append(f"| {m} | {AGG[m]} | **{f_adopt:.3f}** | {f_max:.3f} | {baseB[m]:.3f} |")
    L.append("")

    # タイミング
    L.append("### タイミング（採用 readout, test）")
    L.append("| event | 方式 | det | MAE(s) | ±0.1 | ±0.3 | ±0.5 | ±1.0 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for etype in ["shift", "shift_ov"]:
        tag, m = best_timing(tim_val[etype], tim_test[etype])
        w = m["within"]
        L.append(f"| {etype} | {tag} | {m['det']*100:.0f}% | **{m['MAE']:.3f}** | "
                 f"{w[0.1]*100:.1f}% | {w[0.3]*100:.1f}% | {w[0.5]*100:.1f}% | {w[1.0]*100:.1f}% |")
    L.append("")
    L.append("参考ベースライン: 固定ギャップ 沈黙MAE 0.353 / VAP-onset 沈黙0.635・overlap0.754。")

    out = Path("reports/eval_final.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print("saved ->", out)


if __name__ == "__main__":
    main()
