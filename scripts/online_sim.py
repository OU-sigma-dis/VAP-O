"""疑似オンライン評価: 真の onset を一切使わない時系列シミュレーション。

役割: 査読指摘「onset 局在化評価が true onset でアンカーされた offline 診断であり、
      オンライン性能ではない」に応える。モデルは因果的（フレーム t の出力は t 以前の
      入力のみに依存; ALiBi causal mask）なので、窓ごとの出力を時系列に走査すれば
      同一文脈のストリーミング実行と等価になる。

プロトコル（真の onset 非依存）:
  - 各話者 s を独立に監視。s が沈黙中（VAD、実運用の VAD モジュールの代替として
    GT VAD を使用）に信号が θ を上方交差したら「s がまもなく話し始める」と発火。
    ヒステリシス: 信号が θ-0.15 を下回るまで再発火しない。監視開始は 3s（min context）。
  - 発火の読み出し 2 種:
      react  : 発火時刻そのものを予告時刻とする
      extrap : ラベルの意味論から t_hat = t_fire + (1 - sig(t_fire)) * H を予告時刻とする
  - 照合: 発火は、その話者の実 onset が [onset-2s, onset+0.5s] 内にあれば的中(TP)。
    同一 onset への 2 発目以降は再確認として無視。窓内に対応 onset が無い発火は FA。
    主要 onset（先行 0.5s 以上の沈黙をもつ立ち上がり）で発火が無いものは miss。
    先行沈黙 0.5s 未満の細かい再開への発火は minor 的中（FA にも timing にも数えない）。
  - しきい値 θ は val の onset 検出 F1（recall は主要 onset, precision は全発火）で選択。

比較系:
  - VAP-O (react / extrap)
  - ベースライン VAP p_now (react; p_now に時間の意味論は無いので extrap 不可)
  - 固定ギャップ: 相手話者のオフセット観測 + G=0.55s に発火を予約、相手が予約前に
    再開したらキャンセル（G は val ギャップ中央値、モデル不要）

出力: reports/online_sim.md（キャッシュ reports/_online_cache.pkl）
使い方: .venv/bin/python scripts/online_sim.py
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
from tier1b_paired import VAPO_CKPT, BASEB_CKPT
from utilities.paths import switchboard_paths

FRAME_HZ = 20
H_FRAMES = 60            # H=3.0s
START = 60               # 監視開始 3s（min context と一致）
HYST = 0.15              # ヒステリシス幅
MATCH_PRE = 40           # onset の 2s 前から
MATCH_POST = 10          # onset の 0.5s 後まで
MAJOR_SIL = 10           # 主要 onset の先行沈黙 0.5s
G_FRAMES = 11            # 固定ギャップ 0.55s（val 中央値）
TOLS = [0.1, 0.3, 0.5, 1.0]
SEED = 42
_, VAL_CSV, TEST_CSV, CPC_DIR = switchboard_paths()
CACHE = Path("reports/_online_cache.pkl")


# ---------- 推論（窓ごとの op / p_now / va を収集） ----------

def collect(vapo, baseb, objective, loader):
    """各 20s 窓の op(T,2)・p_now(T,2)・va(T,2) を集める。出力: 窓の辞書リスト。"""
    out = []
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
            op = vo["onset_proximity"].cpu().numpy().astype(np.float16)
            bo = baseb.forward_from_cpc_features(batch.get("cpc_feat_1"), batch.get("cpc_feat_2"))
            probs = bo["logits"].softmax(dim=-1).cpu()
            pnow = objective.probs_next_speaker_aggregate(probs, 0, 1).numpy().astype(np.float16)
            va = (batch["va"].cpu().numpy() > 0.5).astype(np.uint8)
            T = min(op.shape[1], va.shape[1], pnow.shape[1])
            for b in range(op.shape[0]):
                out.append(dict(op=op[b, :T], pnow=pnow[b, :T], va=va[b, :T]))
    return out


# ---------- シミュレーション部品 ----------

def onsets_of(va_s):
    """VAD 立ち上がり一覧。出力: (全 onset, 主要 onset=先行0.5s沈黙) のフレームリスト。"""
    rise = np.where((va_s[1:] == 1) & (va_s[:-1] == 0))[0] + 1
    rise = [int(t) for t in rise if t >= START]
    major = [t for t in rise if t >= MAJOR_SIL and va_s[t - MAJOR_SIL:t].sum() == 0]
    return rise, major


def fires_of(sig_s, va_s, theta):
    """沈黙中の θ 上方交差（ヒステリシス付き）。出力: 発火フレームのリスト。"""
    fires, armed = [], True
    for t in range(START, len(sig_s)):
        if va_s[t] == 1:
            armed = True
            continue
        if armed and sig_s[t] >= theta:
            fires.append(t)
            armed = False
        elif not armed and sig_s[t] < theta - HYST:
            armed = True
    return fires


def fixed_gap_fires(va, s):
    """固定ギャップ: 相手のオフセット+G に発火予約、相手が予約前に再開したらキャンセル。"""
    o = 1 - s
    fires = []
    T = len(va)
    offs = np.where((va[1:, o] == 0) & (va[:-1, o] == 1))[0] + 1
    for t in offs:
        if t < START:
            continue
        sched = t + G_FRAMES
        if sched >= T:
            continue
        if va[t:sched, o].sum() > 0:  # 相手が予約前に再開 → キャンセル
            continue
        fires.append(int(sched))
    return fires


def match_fires(fires, pred_times, all_on, major_on):
    """発火と onset の照合。出力: dict(TP=[(err_s, onset)], FA数, miss数, minor数)。"""
    major_set = set(major_on)
    claimed = {}
    TP, fa = [], 0
    minor = 0
    for tf, tp_ in zip(fires, pred_times):
        cands = [to for to in all_on if to - MATCH_PRE <= tf <= to + MATCH_POST]
        if not cands:
            fa += 1
            continue
        to = min(cands, key=lambda x: abs(x - tf))
        if to in claimed:
            continue  # 同一 onset への再確認は無視
        claimed[to] = tf
        if to in major_set:
            TP.append(((tp_ - to) / FRAME_HZ, to))
        else:
            minor += 1
    miss = sum(1 for to in major_on if to not in claimed)
    return dict(TP=TP, FA=fa, miss=miss, minor=minor)


def run_system(windows, system, theta):
    """1 システムを全窓・全話者で走らせ集計。出力: 指標 dict。"""
    agg = dict(TP=[], FA=0, miss=0, minor=0, n_major=0, tp_onsets=[])
    for w in windows:
        va = w["va"]
        for s in (0, 1):
            all_on, major_on = onsets_of(va[:, s])
            agg["n_major"] += len(major_on)
            if system == "fixed_gap":
                fires = fixed_gap_fires(va, s)
                preds = fires  # 予約時刻そのものが予告時刻
            else:
                sig = w["op"][:, s].astype(np.float32) if system.startswith("vapo") \
                    else w["pnow"][:, s].astype(np.float32)
                fires = fires_of(sig, va[:, s], theta)
                if system == "vapo_extrap":
                    preds = [t + (1.0 - float(sig[t])) * H_FRAMES for t in fires]
                else:
                    preds = fires
            m = match_fires(fires, preds, all_on, major_on)
            agg["TP"] += [e for e, _ in m["TP"]]
            agg["FA"] += m["FA"]
            agg["miss"] += m["miss"]
            agg["minor"] += m["minor"]
    minutes = len(windows) * 20.0 / 60.0
    tp_n = len(agg["TP"]) + agg["minor"]
    prec = tp_n / (tp_n + agg["FA"]) if tp_n + agg["FA"] else 0.0
    rec = len(agg["TP"]) / (len(agg["TP"]) + agg["miss"]) if agg["TP"] or agg["miss"] else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    errs = np.array(agg["TP"])
    res = dict(theta=theta, prec=prec, rec=rec, f1=f1,
               fa_per_min=agg["FA"] / minutes, n_tp=len(agg["TP"]),
               n_major=agg["n_major"], minor=agg["minor"])
    if len(errs):
        ae = np.abs(errs)
        res.update(bias=float(errs.mean()), mae=float(ae.mean()),
                   within={t: float((ae <= t).mean()) for t in TOLS})
    else:
        res.update(bias=np.nan, mae=np.nan, within={t: 0.0 for t in TOLS})
    return res


def shift_subset(windows, ee, theta, system="vapo_react"):
    """shift イベント（沈黙/重複経由）の onset に照合された発火のみの誤差。
    イベント抽出は評価定義の再利用であり、発火規則自体は onset を知らない。"""
    errs = {"shift": [], "shift_ov": []}
    n_ev = {"shift": 0, "shift_ov": 0}
    for w in windows:
        va = torch.from_numpy(w["va"].astype(np.float32)).unsqueeze(0)
        ev = ee(va)
        targets = {"shift": {}, "shift_ov": {}}
        for et in ("shift", "shift_ov"):
            if et not in ev:
                continue
            for a, b_, spk in ev[et][0]:
                # 真の onset: shift=b_(onset_start), shift_ov=a(重複開始)
                to = int(b_) if et == "shift" else int(a)
                targets[et][(int(spk), to)] = True
                n_ev[et] += 1
        for s in (0, 1):
            sig = w["op"][:, s].astype(np.float32)
            all_on, major_on = onsets_of(w["va"][:, s])
            fires = fires_of(sig, w["va"][:, s], theta)
            if system == "vapo_extrap":
                preds = [t + (1.0 - float(sig[t])) * H_FRAMES for t in fires]
            else:
                preds = fires
            claimed = set()
            for tf, tp_ in zip(fires, preds):
                cands = [to for to in all_on if to - MATCH_PRE <= tf <= to + MATCH_POST]
                if not cands:
                    continue
                to = min(cands, key=lambda x: abs(x - tf))
                if to in claimed:
                    continue
                claimed.add(to)
                for et in ("shift", "shift_ov"):
                    for (spk, tev) in targets[et]:
                        if spk == s and abs(tev - to) <= 2:
                            errs[et].append((tp_ - tev) / FRAME_HZ)
    return errs, n_ev


def fmt_row(name, r):
    w = r["within"]
    return (f"| {name} | {r['prec']*100:.1f}% | {r['rec']*100:.1f}% | {r['f1']:.3f} | "
            f"{r['fa_per_min']:.2f} | {r['bias']:+.3f} | {r['mae']:.3f} | "
            f"{w[0.1]*100:.1f}% | {w[0.3]*100:.1f}% | {w[0.5]*100:.1f}% |")


def main():
    os.chdir(_ROOT)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    if CACHE.exists():
        data = pickle.load(open(CACHE, "rb"))
        wins_val, wins_test = data["val"], data["test"]
        print(f"loaded cache: val={len(wins_val)} test={len(wins_test)} windows")
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

        print("collect val...")
        wins_val = collect(vapo, baseb, objective, loader(VAL_CSV))
        print("collect test...")
        wins_test = collect(vapo, baseb, objective, loader(TEST_CSV))
        pickle.dump({"val": wins_val, "test": wins_test,
                     "meta": {"vapo": VAPO_CKPT, "baseb": BASEB_CKPT}}, open(CACHE, "wb"))

    thetas = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    L = ["# 疑似オンライン評価（真の onset 非依存の時系列シミュレーション）\n",
         "- 発火規則: 話者が沈黙中に信号が θ を上方交差（ヒステリシス 0.15, 監視開始 3s）。",
         "  真の onset は発火にも探索窓にも一切使わない。VAD は実運用の VAD モジュールの代替として GT を使用。",
         "- 照合: 発火 ∈ [onset−2s, onset+0.5s] で的中。同一 onset への再発火は無視。対応 onset の無い発火は FA。",
         "  主要 onset（先行 0.5s 以上の沈黙）で発火無し = miss。先行沈黙 0.5s 未満の再開への発火は minor（FA/timing に不算入）。",
         "- 誤差 = 予告時刻 − 真 onset（負 = 早い）。react = 発火時刻, extrap = 発火時刻 + (1−信号)·H。",
         f"- θ は val の F1 で選択。固定ギャップは相手オフセット+0.55s 予約・再開でキャンセル。\n"]

    # === val θ sweep（VAP-O react） ===
    L.append("## val θ スイープ（VAP-O, react）")
    L.append("| θ | 適合率 | 再現率 | F1 | FA/分 | バイアス(s) | MAE(s) | ±0.1 | ±0.3 | ±0.5 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    best = None
    for th in thetas:
        r = run_system(wins_val, "vapo_react", th)
        L.append(fmt_row(f"θ={th}", r))
        if best is None or r["f1"] > best[1]["f1"]:
            best = (th, r)
    th_vapo = best[0]
    L.append(f"\n選択: θ={th_vapo}（val F1 最大）\n")

    # ベースライン p_now の θ も val で選択
    best_b = None
    for th in thetas:
        r = run_system(wins_val, "pnow_react", th)
        if best_b is None or r["f1"] > best_b[1]["f1"]:
            best_b = (th, r)
    th_pnow = best_b[0]

    # === test 本評価（運用点カーブ: 検出重視 θ から タイミング重視 θ まで） ===
    L.append(f"## test 結果（運用点カーブ; val F1 最大は VAP-O θ={th_vapo}, p_now θ={th_pnow}）")
    L.append("| システム | 適合率 | 再現率 | F1 | FA/分 | バイアス(s) | MAE(s) | ±0.1 | ±0.3 | ±0.5 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    op_thetas = [0.6, 0.8, 0.9, 0.95]
    for th in op_thetas:
        r = run_system(wins_test, "vapo_react", th)
        mark = "★" if th == th_vapo else ""
        L.append(fmt_row(f"VAP-O react θ={th}{mark}", r))
    for th in op_thetas:
        r = run_system(wins_test, "vapo_extrap", th)
        L.append(fmt_row(f"VAP-O extrap θ={th}", r))
    for th in sorted({th_pnow, 0.9}):
        r = run_system(wins_test, "pnow_react", th)
        L.append(fmt_row(f"VAP p_now react θ={th}", r))
    r_fg = run_system(wins_test, "fixed_gap", 0.0)
    L.append(fmt_row("固定ギャップ", r_fg))
    r_ref = run_system(wins_test, "vapo_react", th_vapo)
    L.append(f"\n- 主要 onset 数(test, 両話者, 全種): {r_ref['n_major']}。"
             "再現率の分母は全主要 onset（ターン内再開含む）。\n")

    # === shift イベント部分集合（オフライン表との対応, 2 運用点） ===
    ee = TurnTakingEvents(EventConfig())
    L.append("## shift イベント部分集合のタイミング（オンライン発火, オフライン表と対応）")
    L.append("| 系 | θ | 経路 | 検出/総数 | バイアス(s) | MAE(s) | ±0.1 | ±0.3 | ±0.5 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for th in (th_vapo, 0.9):
        for sysname in ("vapo_react", "vapo_extrap"):
            errs, n_ev = shift_subset(wins_test, ee, th, sysname)
            for et, label in [("shift", "沈黙経由"), ("shift_ov", "重複経由")]:
                e = np.array(errs[et])
                if len(e):
                    ae = np.abs(e)
                    L.append(f"| {sysname} | {th} | {label} | {len(e)}/{n_ev[et]} | {e.mean():+.3f} | "
                             f"{ae.mean():.3f} | {(ae<=0.1).mean()*100:.1f}% | "
                             f"{(ae<=0.3).mean()*100:.1f}% | {(ae<=0.5).mean()*100:.1f}% |")
                else:
                    L.append(f"| {sysname} | {th} | {label} | 0/{n_ev[et]} | - | - | - | - | - |")

    out = Path("reports/online_sim.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L)); print("\nsaved ->", out)


if __name__ == "__main__":
    main()
