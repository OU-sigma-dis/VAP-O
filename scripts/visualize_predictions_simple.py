"""予測結果の可視化（簡易・大ラベル版）。

役割: 論文 Figure 3 用に、shift イベント周辺の onset proximity を
      「必要な情報だけ」に絞って見やすく描く。次話者の GT/Pred、沈黙区間、
      true onset、predicted onset（ピーク）とタイミング誤差のみを表示し、
      相手話者の曲線・VAD サブプロット・閾値交差線などは省く。
入力: 学習済みチェックポイント、test データ。
出力: output_dir に shift_*.png（単一パネル、フォント大）。
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import VapConfig, EventConfig
from config.training import OptConfig, DataConfig
from datasets.dataset import VapDataset
from trains.train import VAPModel
from vap.events import TurnTakingEvents
from utilities.paths import switchboard_paths

FRAME_HZ = 20
_, _, TEST_CSV, CPC_DIR = switchboard_paths()

# フォント・線の既定値を大きめに
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})


def moving_average(signal, window):
    """因果的移動平均（タイミング評価と同じ平滑化）。
    入力: 1D 配列, 窓長; 出力: 同長の平滑化配列。"""
    if window <= 1:
        return signal
    pad = np.concatenate([np.full(window - 1, signal[0]), signal])
    kernel = np.ones(window) / window
    return np.convolve(pad, kernel, mode="valid")


def find_peak_after(signal, start, end):
    """start 以降の大域ピーク(argmax)を返す。onset proximity は設計上 onset で
    最大になるため、採用 readout（argmax）と一致させる。"""
    if end <= start + 1:
        return start
    return start + int(np.argmax(signal[start:end]))


def plot_shift(ax, pred_onset, gt_onset, va, speaker, sil_start, onset_start,
               ps_start, ps_end, threshold, window_pad=40, ma=1,
               band_end=None, band_label="silence", band_color="orange",
               band_alpha=0.22, draw_marker=True, show_error=True):
    """1 つの shift イベントを単一パネルで描く（次話者のみ、予測は MA 平滑化）。
    背景に各話者の発話区間を薄く塗る。band_* で沈黙帯/オーバーラップ帯を切替。
    draw_marker=False で予測onsetマーカーを省略（オーバーラップは局在化採点をしないため）。"""
    n_frames = len(pred_onset)
    other = 1 - speaker
    pred = moving_average(pred_onset[:, speaker], ma)  # 表示・ピーク検出用に平滑化
    if band_end is None:
        band_end = onset_start
    w_start = max(0, ps_start - window_pad)
    w_end = min(n_frames, max(onset_start, band_end) + window_pad)
    frames = np.arange(w_start, w_end)
    times = frames / FRAME_HZ
    ylo, yhi = -0.05, 1.12

    # 発話区間を薄く塗る。重なりを避けるため、相手話者=上半分(赤系),
    # 次話者=下半分(青系) に塗り分ける。
    mid = (ylo + yhi) / 2
    ax.fill_between(times, mid, yhi, where=va[w_start:w_end, other] > 0.5,
                    step="mid", color="#f4a6a6", alpha=0.5, lw=0,
                    label="outgoing speaker speech", zorder=0)
    ax.fill_between(times, ylo, mid, where=va[w_start:w_end, speaker] > 0.5,
                    step="mid", color="#9ec5f0", alpha=0.5, lw=0,
                    label="incoming speaker speech", zorder=0)
    ax.axhline(y=mid, color="#e2e8f0", lw=0.8, zorder=0)  # 上下半分の境界

    # 帯: 沈黙 [sil_start, onset) または オーバーラップ [onset, band_end)
    ax.axvspan(sil_start / FRAME_HZ, band_end / FRAME_HZ,
               alpha=band_alpha, color=band_color, label=band_label, zorder=0)

    # GT / Pred（次話者のみ）
    ax.plot(times, gt_onset[w_start:w_end, speaker], color="#08306b",
            lw=2.8, label="ground truth")
    ax.plot(times, pred[w_start:w_end], color="#b30000",
            lw=2.8, ls="--", label="predicted")

    # true onset
    ax.axvline(x=onset_start / FRAME_HZ, color="black", lw=2.4,
               label="true onset")

    # predicted onset（ピーク検出）
    seg = pred[ps_start:ps_end]
    crossing_abs = None
    if seg[0] >= threshold:
        crossing_abs = ps_start
    else:
        for i in range(1, len(seg)):
            if seg[i] >= threshold and seg[i - 1] < threshold:
                crossing_abs = ps_start + i
                break
    error = None
    if draw_marker and crossing_abs is not None:
        search_end = min(n_frames, onset_start + int(1.0 * FRAME_HZ))
        peak = find_peak_after(pred, crossing_abs, search_end)
        predicted_onset = peak + 1
        error = (predicted_onset - onset_start) / FRAME_HZ
        lbl = (f"predicted onset (error {error:+.2f} s)" if show_error
               else "predicted onset (peak)")
        ax.plot(peak / FRAME_HZ, pred[peak], "v",
                color="#9467bd", markersize=15, zorder=6, label=lbl)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Onset proximity")
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlim(times[0], times[-1])
    # 凡例はプロット外（上）に水平配置してデータを隠さない
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, fontsize=12, framealpha=0.9, handlelength=1.5,
              columnspacing=1.1, handletextpad=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    return error


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="runs_evaluation/visualizations_simple")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--event_type", type=str, default="shift",
                        choices=["shift", "shift_ov"])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.serialization.add_safe_globals([VapConfig, OptConfig, DataConfig, EventConfig])
    model = VAPModel.load_from_checkpoint(args.checkpoint, map_location=args.device)
    model.event_extractor = TurnTakingEvents(EventConfig())
    model.eval()
    model = model.float()

    dset = VapDataset(
        path=TEST_CSV,
        window_size=20.0,
        stride=20.0,
        cpc_feature_dir=CPC_DIR,
    )
    loader = DataLoader(dset, batch_size=4, num_workers=0, shuffle=False)

    samples = []
    with torch.no_grad():
        for batch in loader:
            if len(samples) >= args.num_samples:
                break
            for k in batch:
                if isinstance(batch[k], torch.Tensor) and batch[k].is_floating_point():
                    batch[k] = batch[k].float()
            n_frames = batch["onset_proximity"].shape[1]
            out = model(
                audio=batch["waveform"],
                text_tokens=batch["text_tokens"],
                text_token_positions=batch["text_token_positions"],
                n_frames=n_frames,
                cpc_feat_1=batch.get("cpc_feat_1"),
                cpc_feat_2=batch.get("cpc_feat_2"),
            )
            events = model.event_extractor(batch["va"])
            B = batch["va"].shape[0]
            pk = "pred_" + args.event_type
            ek = args.event_type
            for b in range(B):
                if len(samples) >= args.num_samples:
                    break
                if pk not in events or ek not in events:
                    continue
                for ps_start, ps_end, ps_speaker in events[pk][b]:
                    if len(samples) >= args.num_samples:
                        break
                    best = None
                    for sil_start, onset_start, sh_speaker in events[ek][b]:
                        if sh_speaker != ps_speaker:
                            continue
                        dist = abs(ps_end - sil_start)
                        if dist <= 5 and (best is None or dist < best[0]):
                            best = (dist, sil_start, onset_start)
                    if best is None:
                        continue
                    _, sil_start, onset_start = best
                    samples.append({
                        "pred_onset": out["onset_proximity"][b].cpu().numpy(),
                        "gt_onset": batch["onset_proximity"][b].cpu().numpy(),
                        "va": batch["va"][b].cpu().numpy(),
                        "speaker": ps_speaker,
                        "sil_start": sil_start,
                        "onset_start": onset_start,
                        "ps_start": ps_start,
                        "ps_end": ps_end,
                        "session": batch["session"][b],
                    })

    print(f"Collected {len(samples)} samples")
    for i, s in enumerate(samples):
        fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.0))
        if args.event_type == "shift_ov":
            err = plot_shift(
                ax,
                pred_onset=s["pred_onset"], gt_onset=s["gt_onset"], va=s["va"],
                speaker=s["speaker"], sil_start=s["sil_start"],
                onset_start=s["sil_start"], ps_start=s["ps_start"],
                ps_end=s["ps_end"], threshold=args.threshold,
                band_end=s["onset_start"], band_label="overlap",
                band_color="#b19cd9", draw_marker=False,
            )
        else:
            err = plot_shift(
                ax,
                pred_onset=s["pred_onset"], gt_onset=s["gt_onset"], va=s["va"],
                speaker=s["speaker"], sil_start=s["sil_start"],
                onset_start=s["onset_start"], ps_start=s["ps_start"],
                ps_end=s["ps_end"], threshold=args.threshold,
            )
        fig.tight_layout()
        fp = output_dir / f"{args.event_type}_simple_{i:02d}_{s['session']}.png"
        fig.savefig(fp, dpi=160, bbox_inches="tight")
        plt.close(fig)
        es = f"{err:+.2f}s" if err is not None else "no-marker"
        print(f"  Saved: {fp.name}  (error={es})")
    print(f"Done. {len(samples)} figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
