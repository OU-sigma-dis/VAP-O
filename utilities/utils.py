import os
from os.path import dirname
from typing import List, Tuple

import torch
from torch import Tensor

from utilities.audio import time_to_frames


def repo_root():
    """
    Returns the absolute path to the git repository
    """
    root = dirname(__file__)
    root = dirname(root)
    return root


def get_default_device():
    """利用可能な最適なデバイスを取得"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def everything_deterministic():
    """
    -----------------------------
    Wav2Vec
    -------
    1. Settings
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(mode=True)
    2. Load Model
    3. backprop from step and plot

    RuntimeError: replication_pad1d_backward_cuda does not have a deterministic
    implementation, but you set 'torch.use_deterministic_algorithms(True)'. You can
    turn off determinism just for this operation if that's acceptable for your
    application. You can also file an issue at
    https://github.com/pytorch/pytorch/issues to help us prioritize adding
    deterministic support for this operation.


    -----------------------------
    CPC
    -------
    1. Settings
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(mode=True)
    2. Load Model
    3. backprop from step and plot

    RuntimeError: Deterministic behavior was enabled with either
    `torch.use_deterministic_algorithms(True)` or
    `at::Context::setDeterministicAlgorithms(true)`, but this operation is not
    deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable
    deterministic behavior in this case, you must set an environment variable
    before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or
    CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to
    https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility


    Set these ENV variables and it works with the above recipe

    bash:
        export CUBLAS_WORKSPACE_CONFIG=:4096:8
        export CUBLAS_WORKSPACE_CONFIG=:16:8

    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True)


def find_island_idx_len(
    x: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Finds patches of the same value.

    starts_idx, duration, values = find_island_idx_len(x)

    e.g:
        ends = starts_idx + duration

        s_n = starts_idx[values==n]
        ends_n = s_n + duration[values==n]  # find all patches with N value

    """
    assert x.ndim == 1
    n = len(x)
    y = x[1:] != x[:-1]  # pairwise unequal (string safe)
    i = torch.cat(
        (torch.where(y)[0], torch.tensor(n - 1, device=x.device).unsqueeze(0))
    ).long()
    it = torch.cat((torch.tensor(-1, device=x.device).unsqueeze(0), i))
    dur = it[1:] - it[:-1]
    idx = torch.cumsum(
        torch.cat((torch.tensor([0], device=x.device, dtype=torch.long), dur)), dim=0
    )[:-1]  # positions
    return idx, dur, x[i]


def vad_fill_silences(
    vad: Tensor, max_fill_time: float = 0.02, frame_hz: float = 20
) -> Tensor:
    assert vad.ndim == 2, f"Expects (N_FRAMES, 2) got {vad.shape}"
    assert vad.shape[-1] == 2, f"Expects (N_FRAMES, 2) got {vad.shape}"
    max_fill_frame = round(max_fill_time * frame_hz)
    for ch in range(2):
        starts, dur, on_off = find_island_idx_len(vad[:, ch])
        sil_starts = starts[on_off == 0]
        sil_durs = dur[on_off == 0]
        w = torch.where(sil_durs <= max_fill_frame)[0]
        fill_starts = sil_starts[w]
        fill_durs = sil_durs[w]
        for s, d in zip(fill_starts, fill_durs):
            vad[s : s + d, ch] = 1.0
    return vad


def vad_omit_spikes(
    vad: Tensor, max_omit_time: float = 0.02, frame_hz: float = 20
) -> Tensor:
    assert vad.ndim == 2, f"Expects (N_FRAMES, 2) got {vad.shape}"
    assert vad.shape[-1] == 2, f"Expects (N_FRAMES, 2) got {vad.shape}"
    max_omit_frame = round(max_omit_time * frame_hz)
    for ch in range(2):
        starts, dur, on_off = find_island_idx_len(vad[:, ch])
        sil_starts = starts[on_off == 1]
        sil_durs = dur[on_off == 1]
        w = torch.where(sil_durs <= max_omit_frame)[0]
        fill_starts = sil_starts[w]
        fill_durs = sil_durs[w]
        for s, d in zip(fill_starts, fill_durs):
            vad[s : s + d, ch] = 0.0
    return vad


def vad_fill_silences_batch(
    vad_batch: Tensor, max_fill_time: float = 0.02, frame_hz: float = 20
) -> Tensor:
    """
    バッチ処理対応版のvad_fill_silences

    Args:
        vad_batch: (B, N_FRAMES, 2) の形状のテンソル
        max_fill_time: 埋める最大の無音時間（秒）
        frame_hz: フレームレート

    Returns:
        処理済みのVADテンソル (B, N_FRAMES, 2)
    """
    assert vad_batch.ndim == 3, f"Expects (B, N_FRAMES, 2) got {vad_batch.shape}"
    assert vad_batch.shape[-1] == 2, f"Expects (B, N_FRAMES, 2) got {vad_batch.shape}"

    batch_size = vad_batch.shape[0]
    result = vad_batch.clone()

    for b in range(batch_size):
        result[b] = vad_fill_silences(result[b], max_fill_time, frame_hz)

    return result


def vad_omit_spikes_batch(
    vad_batch: Tensor, max_omit_time: float = 0.02, frame_hz: float = 20
) -> Tensor:
    """
    バッチ処理対応版のvad_omit_spikes

    Args:
        vad_batch: (B, N_FRAMES, 2) の形状のテンソル
        max_omit_time: 除去する最大のスパイク時間（秒）
        frame_hz: フレームレート

    Returns:
        処理済みのVADテンソル (B, N_FRAMES, 2)
    """
    assert vad_batch.ndim == 3, f"Expects (B, N_FRAMES, 2) got {vad_batch.shape}"
    assert vad_batch.shape[-1] == 2, f"Expects (B, N_FRAMES, 2) got {vad_batch.shape}"

    batch_size = vad_batch.shape[0]
    result = vad_batch.clone()

    for b in range(batch_size):
        result[b] = vad_omit_spikes(result[b], max_omit_time, frame_hz)

    return result


def vad_list_to_onehot(
    vad_list: List[List[List[float]]],
    duration: float,
    hop_time: float = 0,
    frame_hz: float = 0,
    channel_first: bool = False,
) -> Tensor:
    assert hop_time > 0 or frame_hz > 0, (
        "vad_list_to_onehot requires `frame_hz` or `hop_time`"
    )

    if frame_hz > 0:
        hop_time = 1 / frame_hz

    n_frames = time_to_frames(duration, hop_time)
    vad_tensor = torch.zeros((n_frames, 2), dtype=torch.float32)

    # ベクトル化処理で高速化
    for ch, ch_vad in enumerate(vad_list):
        if ch_vad:  # 空でない場合のみ処理
            # 全ての開始・終了時刻を一度に変換
            starts_ends = torch.tensor(ch_vad, dtype=torch.float32)
            starts = (starts_ends[:, 0] / hop_time).long()
            ends = (starts_ends[:, 1] / hop_time).long()

            # フレーム境界をクリップ
            starts = torch.clamp(starts, 0, n_frames - 1)
            ends = torch.clamp(ends, 0, n_frames)

            # 高速化：有効な範囲のみフィルタリング
            valid_mask = starts < ends
            valid_starts = starts[valid_mask]
            valid_ends = ends[valid_mask]

            # ベクトル化された範囲設定
            for s, e in zip(valid_starts, valid_ends):
                vad_tensor[s:e, ch] = 1.0

    if channel_first:
        vad_tensor = vad_tensor.permute(1, 0)

    return vad_tensor


def vad_list_to_onehot_fast(
    vad_list: List[List[List[float]]],
    duration: float,
    hop_time: float = 0,
    frame_hz: float = 0,
    channel_first: bool = False,
) -> Tensor:
    """
    更に高速化されたVAD変換関数 - インデックスベースの高速処理
    """
    assert hop_time > 0 or frame_hz > 0, (
        "vad_list_to_onehot_fast requires `frame_hz` or `hop_time`"
    )

    if frame_hz > 0:
        hop_time = 1 / frame_hz

    n_frames = time_to_frames(duration, hop_time)
    vad_tensor = torch.zeros((n_frames, 2), dtype=torch.float32)

    # チャンネルごとに一括処理
    for ch, ch_vad in enumerate(vad_list):
        if ch_vad:
            # 全区間を一度にテンソル化
            intervals = torch.tensor(ch_vad, dtype=torch.float32)
            starts = torch.clamp((intervals[:, 0] / hop_time).long(), 0, n_frames - 1)
            ends = torch.clamp((intervals[:, 1] / hop_time).long(), 0, n_frames)

            # 有効な区間のみ抽出
            valid_mask = starts < ends
            if valid_mask.any():
                valid_starts = starts[valid_mask]
                valid_ends = ends[valid_mask]

                # インデックスを作成して一括設定
                for s, e in zip(valid_starts, valid_ends):
                    vad_tensor[s:e, ch] = 1.0

    if channel_first:
        vad_tensor = vad_tensor.permute(1, 0)

    return vad_tensor


if __name__ == "__main__":
    # サンプルVADデータ
    vad_example = [
        [[0.0, 0.5], [1.0, 1.5]],  # チャンネル1の発話区間
        [[0.2, 0.7], [1.2, 1.7]],  # チャンネル2の発話区間
    ]
    duration = 2.0  # 音声の総時間（秒）
    hop_time = 0.01  # フレームの時間間隔（秒）

    vad_tensor = vad_list_to_onehot(vad_example, duration, hop_time=hop_time)
    print("VAD Tensor:")
    print(vad_tensor)

    # vad_fill_silencesのテスト
    vad_with_silences = torch.tensor(
        [
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 1],
            [0, 0],
            [1, 1],
            [0, 0],
            [0, 0],
            [1, 0],
        ],
        dtype=torch.float32,
    )
    print("\nOriginal VAD with silences:")
    print(vad_with_silences)

    filled_vad = vad_fill_silences(vad_with_silences, max_fill_time=0.02, frame_hz=100)
    print("\nFilled VAD:")
    print(filled_vad)

    # vad_omit_spikesのテスト
    vad_with_spikes = torch.tensor(
        [
            [1, 1],
            [0, 1],
            [1, 1],
            [1, 0],
            [1, 1],
            [0, 0],
            [1, 1],
            [1, 1],
            [0, 1],
        ],
        dtype=torch.float32,
    )
    print("\nOriginal VAD with spikes:")
    print(vad_with_spikes)

    omitted_vad = vad_omit_spikes(vad_with_spikes, max_omit_time=0.02, frame_hz=100)
    print("\nOmitted VAD:")
    print(omitted_vad)
