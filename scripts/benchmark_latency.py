"""
オンライン/レイテンシ評価。

役割: VAP-O を実時間ターンテイキングに用いる際の推論レイテンシを測定する。
      20Hz 動作（フレーム周期 50ms）を予算とし、1 会話（batch=1）の 1 ステップ
      あたりの処理時間と実時間係数（RTF）を CPU / MPS で測定する。

測定する 3 つの経路:
  (1) cached  : 事前抽出済み CPC 特徴量を用い、Transformer + (text) + heads を
                400 フレーム窓上で実行（CPC をキャッシュする実運用の逐次コスト）。
  (2) cpc20s  : 20 秒窓・両チャネルの CPC エンコードのみ（キャッシュで回避される部分）。
  (3) full    : 生音声 20 秒から CPC を毎回再計算する end-to-end（最悪上界）。

入力: 学習済みチェックポイント（任意）。重みの有無はレイテンシに影響しないが、
      実機の現実的な構成で測るため可能なら読み込む。
出力: 標準出力に統計、scripts/latency_results.json に結果を保存。
"""
import argparse
import json
import time
from statistics import mean, pstdev

import torch

from config import VapConfig
from models.model import VapGPT


def _strip_ckpt(state):
    """Lightning ckpt の state_dict から VapGPT 用のキーへ整える。
    入力: ckpt 全体 or state_dict
    出力: "model." 等の接頭辞を除いた state_dict"""
    sd = state.get("state_dict", state) if isinstance(state, dict) else state
    out = {}
    for k, v in sd.items():
        nk = k
        for pref in ("model.", "net.", "vap.", "module."):
            if nk.startswith(pref):
                nk = nk[len(pref):]
                break
        out[nk] = v
    return out


def build_model(use_text, ckpt_path, device, use_stereo=False):
    """VapGPT を構築し、可能なら重みを読み込む。
    入力: use_text, ckpt_path(str or None), device, use_stereo
    出力: eval モードのモデル"""
    conf = VapConfig(use_text=use_text, use_stereo=use_stereo)
    model = VapGPT(conf)
    if ckpt_path:
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            missing, unexpected = model.load_state_dict(_strip_ckpt(state), strict=False)
            print(f"  loaded {ckpt_path.split('/')[-1]} "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        except Exception as e:  # 重み不一致でもレイテンシ測定は可能
            print(f"  [warn] could not load weights ({e}); timing uses initialized weights")
    model.eval().to(device)
    return model, conf


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def timeit(fn, device, n_warmup=5, n_iter=50):
    """関数のレイテンシを測定する。
    入力: fn(無引数), device, ウォームアップ回数, 計測回数
    出力: (mean_ms, std_ms, p50_ms, p90_ms) の辞書"""
    for _ in range(n_warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "mean_ms": mean(times),
        "std_ms": pstdev(times),
        "p50_ms": times[len(times) // 2],
        "p90_ms": times[int(len(times) * 0.9)],
        "min_ms": times[0],
    }


def make_inputs(conf, device, window_sec=20):
    """ベンチマーク用のダミー入力を作る（形状は実運用と同一）。
    入力: conf, device, 窓長(秒)
    出力: 入力テンソル群の辞書"""
    n_frames = window_sec * conf.frame_hz          # 400
    n_samples = conf.sample_rate * window_sec      # 320000
    audio = torch.randn(1, 2, n_samples, device=device)
    cpc1 = torch.randn(1, window_sec * 100, 256, device=device)  # 100Hz CPC feats
    cpc2 = torch.randn(1, window_sec * 100, 256, device=device)
    text_tokens = torch.randint(0, conf.vocab_size, (1, conf.max_text_tokens), device=device)
    positions = torch.full((1, conf.max_text_tokens), -1, dtype=torch.long, device=device)
    positions[0, :30] = torch.sort(torch.randint(0, n_frames, (30,)))[0]
    return dict(audio=audio, cpc1=cpc1, cpc2=cpc2, n_frames=n_frames,
                text_tokens=text_tokens, positions=positions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_ckpt", default="output/checkpoints_use_text/"
                    "VapGPT_20Hz_cpc_glove100gru_d256-epoch29-onset_0.24580.ckpt")
    ap.add_argument("--notext_ckpt", default="output/checkpoints_no_text/"
                    "VapGPT_20Hz_cpc_glove100gru_d256-epoch29-onset_0.29334.ckpt")
    ap.add_argument("--window_sec", type=int, default=20)
    ap.add_argument("--n_iter", type=int, default=50)
    ap.add_argument("--stereo_ckpt", default=None,
                    help="指定時: GPT-Stereo 音声のみモデル（最終提案構成）だけを測定")
    args = ap.parse_args()

    devices = ["cpu"]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")

    frame_period_ms = 1000.0 / 20  # 50 ms @ 20Hz
    window_ms = args.window_sec * 1000.0
    n_threads = torch.get_num_threads()
    print(f"torch={torch.__version__}  threads={n_threads}  "
          f"frame_period={frame_period_ms:.1f}ms  window={args.window_sec}s")

    results = {"meta": {"torch": torch.__version__, "threads": n_threads,
                        "frame_period_ms": frame_period_ms,
                        "window_sec": args.window_sec, "n_iter": args.n_iter,
                        "batch": 1}, "runs": []}

    # stereo_ckpt 指定時は最終提案構成（GPT-Stereo, 音声のみ）だけを測る
    if args.stereo_ckpt:
        variants = [("stereo", False, True, args.stereo_ckpt)]
    else:
        variants = [("text", True, False, args.text_ckpt),
                    ("no-text", False, False, args.notext_ckpt)]

    for device in devices:
        for name, use_text, use_stereo, ckpt in variants:
            tag = f"{device}/{name}"
            print(f"\n=== {tag} ===")
            model, conf = build_model(use_text, ckpt, device, use_stereo=use_stereo)
            x = make_inputs(conf, device, args.window_sec)

            @torch.no_grad()
            def run_cached():
                model(x["audio"], x["text_tokens"], x["positions"], x["n_frames"],
                      cpc_feat_1=x["cpc1"], cpc_feat_2=x["cpc2"])

            @torch.no_grad()
            def run_full():
                model(x["audio"], x["text_tokens"], x["positions"], x["n_frames"])

            @torch.no_grad()
            def run_cpc():
                model.audio_encoder(x["audio"][:, :1])
                model.audio_encoder(x["audio"][:, 1:])

            cached = timeit(run_cached, device, n_iter=args.n_iter)
            cpc = timeit(run_cpc, device, n_iter=max(10, args.n_iter // 2))
            full = timeit(run_full, device, n_iter=max(10, args.n_iter // 2))

            row = {
                "device": device, "text": use_text,
                "cached_ms": cached, "cpc20s_ms": cpc, "full_ms": full,
                "rtf_full": full["mean_ms"] / window_ms,
                "rtf_cached": cached["mean_ms"] / window_ms,
                "step_budget_ms": frame_period_ms,
                "realtime_cached": cached["mean_ms"] < frame_period_ms,
            }
            results["runs"].append(row)
            print(f"  cached  (transformer+heads, 400f): {cached['mean_ms']:7.2f} "
                  f"± {cached['std_ms']:.2f} ms  (p90 {cached['p90_ms']:.2f})")
            print(f"  cpc 20s (both channels)          : {cpc['mean_ms']:7.2f} "
                  f"± {cpc['std_ms']:.2f} ms")
            print(f"  full    (re-encode + forward)    : {full['mean_ms']:7.2f} "
                  f"± {full['std_ms']:.2f} ms  -> RTF {row['rtf_full']:.4f}")
            print(f"  recurring step < 50ms budget?    : {row['realtime_cached']}")

    out = "scripts/latency_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
