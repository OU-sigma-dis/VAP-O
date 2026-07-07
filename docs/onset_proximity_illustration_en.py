"""
Intuitive visualization of the Onset Proximity label (English version).

Role: show the waveform produced by report.md's definition
      onset_proximity[t,s] = max(0, 1 - d(t,s)/H), and contrast it with the
      conventional speech-ratio target.
Input: none (synthetic illustrative example)
Output: onset_proximity_illustration_en.png
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

# --- synthetic data for illustration ---
fps = 100
T = 11.0
t = np.linspace(0, T, int(T * fps) + 1)
H = 3.0  # horizon

# Next speaker (speaker B) speech segments; the start of each segment is an onset
b_segments = [(2.0, 3.2), (6.5, 7.6), (9.0, 10.3)]
b_onsets = [s for s, _ in b_segments]

# Current speaker (speaker A) speech segments
a_segments = [(0.0, 1.8), (3.6, 6.2)]


def onset_proximity(time, onsets, H):
    """Proximity from each frame to the next onset.
    Input: time (frame timestamps), onsets (onset times), H (normalization horizon)
    Output: proximity array (0..1); frames with no future onset are 0."""
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


def speech_ratio(time, segments, win=0.5):
    """Proportion of speech within the next `win` seconds.
    Input: time, segments (speech segments), win (window length)
    Output: ratio array (0..1)."""
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

# --- drawing ---
fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True,
                         gridspec_kw={"height_ratios": [1.1, 1.6, 1.6]})
plt.subplots_adjust(hspace=0.38)

cA, cB, cProx, cRatio = "#94a3b8", "#2563eb", "#d97706", "#0d9488"

# (1) speech segments
ax = axes[0]
for s, e in a_segments:
    ax.broken_barh([(s, e - s)], (0.55, 0.32), facecolors=cA, edgecolor="none")
for s, e in b_segments:
    ax.broken_barh([(s, e - s)], (0.12, 0.32), facecolors=cB, edgecolor="none")
for o in b_onsets:
    ax.annotate("onset", xy=(o, 0.44), xytext=(o, 1.15),
                ha="center", fontsize=12, color=cB, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=cB, lw=1.6))
ax.text(-0.15, 0.71, "Speaker A", va="center", ha="right", fontsize=13, color="#475569")
ax.text(-0.15, 0.28, "Speaker B", va="center", ha="right", fontsize=13, color=cB)
ax.set_ylim(0, 1.4)
ax.set_yticks([])
ax.set_title("Speech activity   (the start of each of B's utterances is an onset)",
             fontsize=14, loc="left")

# (2) onset proximity
ax = axes[1]
ax.plot(t, prox_b, color=cProx, lw=2.4)
ax.fill_between(t, prox_b, color=cProx, alpha=0.12)
for o in b_onsets:
    ax.axvline(o, color=cB, ls="--", lw=1.1, alpha=0.7)
    ax.plot(o, 1.0, "o", color=cB, ms=9)
ax.axhline(1.0, color="#cbd5e1", lw=0.8, ls=":")
ax.text(0.05, 1.04, "peak of 1.0 at the onset", fontsize=12, color=cB)
ax.annotate("more than 3 s ahead -> 0", xy=(4.6, 0.0), xytext=(4.6, 0.42),
            fontsize=12, color="#64748b", ha="center",
            arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=1.2))
ax.annotate("rises linearly as the onset nears", xy=(8.3, 0.55), xytext=(6.6, 0.82),
            fontsize=12, color=cProx,
            arrowprops=dict(arrowstyle="->", color=cProx, lw=1.4))
ax.set_ylim(-0.05, 1.25)
ax.set_ylabel("value")
ax.set_title("Proposed: Onset Proximity   (the onset is an explicit peak -> timing is estimable)",
             fontsize=14, loc="left", color=cProx)

# (3) speech ratio
ax = axes[2]
ax.plot(t, ratio_b, color=cRatio, lw=2.4)
ax.fill_between(t, ratio_b, color=cRatio, alpha=0.12)
for o in b_onsets:
    ax.axvline(o, color=cB, ls="--", lw=1.1, alpha=0.7)
pr_start, pr_end = b_onsets[1] - 0.5, b_onsets[1]
ax.axvspan(pr_start, pr_end, color="#bbf7d0", alpha=0.6)
ax.annotate("leading edge of the\nprediction region is\nstructurally low",
            xy=(pr_start + 0.05, 0.15),
            xytext=(4.0, 0.66), fontsize=12, color="#b45309",
            arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.3))
ax.set_ylim(-0.05, 1.25)
ax.set_ylabel("value")
ax.set_xlabel("time (s)")
ax.set_title("Conventional: Speech Ratio   (future N-s speech proportion -> no exact onset position)",
             fontsize=14, loc="left", color=cRatio)

for ax in axes:
    ax.set_xlim(-0.05, T)
    ax.spines[["top", "right"]].set_visible(False)

fig.savefig("/Users/onishi/VAP-O/docs/onset_proximity_illustration_en.png",
            dpi=150, bbox_inches="tight")
print("saved")
