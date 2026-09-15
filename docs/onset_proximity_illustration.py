"""
Onset Proximity ラベルの直感的な可視化。

役割: report.md の式 onset_proximity[t,s] = max(0, 1 - d(t,s)/H) が
      どんな波形になるかを図で示す。比較として従来の speech ratio も描く。
入力: なし（説明用の合成例）
出力: onset_proximity_illustration.png
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

# 日本語表示のため、macOS標準の Hiragino を登録
_jp = "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc"
try:
    font_manager.fontManager.addfont(_jp)
    matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_jp).get_name()
except Exception:
    matplotlib.rcParams["font.family"] = "Hiragino Sans GB"
matplotlib.rcParams["axes.unicode_minus"] = False

# --- 説明用の合成データ ---
fps = 100
T = 11.0
t = np.linspace(0, T, int(T * fps) + 1)
H = 3.0  # horizon

# 次話者(話者B)の発話区間。各区間の先頭が onset
b_segments = [(2.0, 3.2), (6.5, 7.6), (9.0, 10.3)]
b_onsets = [s for s, _ in b_segments]

# 話者A(現話者)の発話区間。Bのonsetまで話している想定
a_segments = [(0.0, 1.8), (3.6, 6.2)]

def onset_proximity(time, onsets, H):
    """各フレームから次のonsetまでの距離で proximity を計算。
    入力: time フレーム時刻配列, onsets onset時刻リスト, H 正規化horizon
    出力: proximity 配列(0..1)。次のonsetが無いフレームは0"""
    prox = np.zeros_like(time)
    for i, tt in enumerate(time):
        future = [o for o in onsets if o >= tt]
        if not future:
            prox[i] = 0.0
            continue
        d = future[0] - tt
        prox[i] = max(0.0, 1.0 - d / H)
    return prox

prox_b = onset_proximity(t, b_onsets, H)

# 従来の speech ratio 風の信号(未来0.5sの発話割合)を簡易再現
def speech_ratio(time, segments, win=0.5):
    """未来 win 秒間に話者が話している割合。
    入力: time, segments 発話区間, win 窓長
    出力: ratio 配列(0..1)"""
    ratio = np.zeros_like(time)
    for i, tt in enumerate(time):
        future = np.linspace(tt, tt + win, 20)
        active = 0
        for f in future:
            if any(s <= f <= e for s, e in segments):
                active += 1
        ratio[i] = active / len(future)
    return ratio

ratio_b = speech_ratio(t, b_segments)

# --- 描画 ---
fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True,
                         gridspec_kw={"height_ratios": [1.1, 1.6, 1.6]})
plt.subplots_adjust(hspace=0.35)

cA, cB, cProx, cRatio = "#94a3b8", "#2563eb", "#d97706", "#0d9488"

# (1) 発話区間
ax = axes[0]
for s, e in a_segments:
    ax.broken_barh([(s, e - s)], (0.55, 0.32), facecolors=cA, edgecolor="none")
for s, e in b_segments:
    ax.broken_barh([(s, e - s)], (0.12, 0.32), facecolors=cB, edgecolor="none")
for o in b_onsets:
    ax.annotate("onset", xy=(o, 0.44), xytext=(o, 1.15),
                ha="center", fontsize=9, color=cB, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=cB, lw=1.6))
ax.text(-0.15, 0.71, "話者A", va="center", ha="right", fontsize=10, color="#475569")
ax.text(-0.15, 0.28, "話者B", va="center", ha="right", fontsize=10, color=cB)
ax.set_ylim(0, 1.4); ax.set_yticks([])
ax.set_title("発話区間  Bの各発話の先頭が onset", fontsize=11, loc="left")

# (2) onset proximity
ax = axes[1]
ax.plot(t, prox_b, color=cProx, lw=2.4)
ax.fill_between(t, prox_b, color=cProx, alpha=0.12)
for o in b_onsets:
    ax.axvline(o, color=cB, ls="--", lw=1.1, alpha=0.7)
    ax.plot(o, 1.0, "o", color=cB, ms=7)
ax.axhline(1.0, color="#cbd5e1", lw=0.8, ls=":")
ax.text(0.05, 1.04, "onsetで 1.0 のピーク", fontsize=9, color=cB)
ax.annotate("3秒以上先 → 0", xy=(4.6, 0.0), xytext=(4.6, 0.42),
            fontsize=9, color="#64748b", ha="center",
            arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.2))
ax.annotate("近づくほど直線的に上昇", xy=(8.3, 0.55), xytext=(7.0, 0.78),
            fontsize=9, color=cProx,
            arrowprops=dict(arrowstyle="->", color=cProx, lw=1.4))
ax.set_ylim(-0.05, 1.25); ax.set_ylabel("値")
ax.set_title("提案 Onset Proximity   onsetが信号のピークとして明示される → タイミング推定が可能",
             fontsize=11, loc="left", color=cProx)

# (3) speech ratio
ax = axes[2]
ax.plot(t, ratio_b, color=cRatio, lw=2.4)
ax.fill_between(t, ratio_b, color=cRatio, alpha=0.12)
for o in b_onsets:
    ax.axvline(o, color=cB, ls="--", lw=1.1, alpha=0.7)
# 予測領域(沈黙直前0.5s)を例示
pr_start, pr_end = b_onsets[1] - 0.5, b_onsets[1]
ax.axvspan(pr_start, pr_end, color="#bbf7d0", alpha=0.6)
ax.annotate("予測領域の先頭は\n構造的に低い", xy=(pr_start + 0.05, 0.15),
            xytext=(4.3, 0.62), fontsize=9, color="#b45309",
            arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.3))
ax.set_ylim(-0.05, 1.25); ax.set_ylabel("値"); ax.set_xlabel("時間 (秒)")
ax.set_title("従来 Speech Ratio   未来N秒の発話割合 → onsetの正確な位置は出ない",
             fontsize=11, loc="left", color=cRatio)

for ax in axes:
    ax.set_xlim(-0.05, T)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Onset Proximity の直感的イメージ", fontsize=14, fontweight="bold", x=0.5)
output_path = Path(__file__).with_name("onset_proximity_illustration.png")
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"saved: {output_path}")
