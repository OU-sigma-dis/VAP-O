from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def bin_times_to_frames(bin_times: List[float], frame_hz: int) -> List[int]:
    return (torch.tensor(bin_times) * frame_hz).long().tolist()


class ProjectionWindow:
    def __init__(
        self,
        bin_times: List = [0.2, 0.4, 0.6, 0.8],
        frame_hz: int = 20,
        threshold_ratio: float = 0.5,
    ):
        super().__init__()
        self.bin_times = bin_times
        self.frame_hz = frame_hz
        self.threshold_ratio = threshold_ratio

        self.bin_frames = bin_times_to_frames(bin_times, frame_hz)
        self.n_bins = len(self.bin_frames)
        self.total_bins = self.n_bins * 2
        self.horizon = sum(self.bin_frames)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(\n"
        s += f"  bin_times: {self.bin_times}\n"
        s += f"  bin_frames: {self.bin_frames}\n"
        s += f"  frame_hz: {self.frame_hz}\n"
        s += f"  thresh: {self.threshold_ratio}\n"
        s += ")\n"
        return s

    def projection(self, va: Tensor) -> Tensor:
        """
        Extract projection (bins)
        (b, n, c) -> (b, N, c, M), M=horizon window size, N=valid frames

        Arguments:
            va:         Tensor (B, N, C)

        Returns:
            vaps:       Tensor (B, m, C, M)

        """
        # Shift to get next frame projections
        return va[..., 1:, :].unfold(dimension=-2, size=sum(self.bin_frames), step=1)

    def projection_bins(self, projection_window: Tensor) -> Tensor:
        """
        Iterate over the bin boundaries and sum the activity
        for each channel/speaker.
        divide by the number of frames to get activity ratio.
        If ratio is greater than or equal to the threshold_ratio
        the bin is considered active
        """

        start = 0
        v_bins = []
        for b in self.bin_frames:
            end = start + b
            m = projection_window[..., start:end].sum(dim=-1) / b
            m = (m >= self.threshold_ratio).float()
            v_bins.append(m)
            start = end
        return torch.stack(v_bins, dim=-1)  # (*, t, c, n_bins)

    def __call__(self, va: Tensor) -> Tensor:
        projection_windows = self.projection(va)
        return self.projection_bins(projection_windows)


class Codebook(nn.Module):
    def __init__(self, bin_frames):
        super().__init__()
        self.bin_frames = bin_frames
        self.n_bins: int = len(self.bin_frames)
        self.total_bins: int = self.n_bins * 2
        self.n_classes: int = 2**self.total_bins

        self.emb = nn.Embedding(
            num_embeddings=self.n_classes, embedding_dim=self.total_bins
        )
        self.emb.weight.data = self.create_code_vectors(self.total_bins)
        self.emb.weight.requires_grad_(False)

    def single_idx_to_onehot(self, idx: int, d: int = 8) -> Tensor:
        assert idx < 2**d, "must be possible with {d} binary digits"
        z = torch.zeros(d)
        b = bin(idx).replace("0b", "")
        for i, v in enumerate(b[::-1]):
            z[i] = float(v)
        return z

    def create_code_vectors(self, n_bins: int) -> Tensor:
        """
        Create a matrix of all one-hot encodings representing a binary sequence of `self.total_bins` places
        Useful for usage in `nn.Embedding` like module.
        """
        n_codes = 2**n_bins
        embs = torch.zeros((n_codes, n_bins))
        for i in range(2**n_bins):
            embs[i] = self.single_idx_to_onehot(i, d=n_bins)
        return embs

    def encode(self, x: Tensor) -> Tensor:
        """

        Encodes projection_windows x (*, 2, 4) to indices in codebook (..., 1)

        Arguments:
            x:          Tensor (*, 2, 4)

        Inspiration for distance calculation:
            https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py
        """
        assert x.shape[-2:] == (
            2,
            self.n_bins,
        ), f"Codebook expects (..., 2, {self.n_bins}) got {x.shape}"

        # compare with codebook and get closest idx
        shape = x.shape
        flatten = rearrange(x, "... c bpp -> (...) (c bpp)", c=2, bpp=self.n_bins)
        embed = self.emb.weight.T
        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        embed_ind = embed_ind.view(*shape[:-2])
        return embed_ind

    def decode(self, idx: Tensor):
        v = self.emb(idx)
        return rearrange(v, "... (c b) -> ... c b", c=2)

    def forward(self, projection_windows: Tensor):
        return self.encode(projection_windows)


class ObjectiveVAP(nn.Module):
    """
    Voice Activity Projection (VAP) の目的関数（損失計算）と評価指標の抽出を管理するクラス。
    """

    def __init__(
        self,
        bin_times: list[float] = [0.2, 0.4, 0.6, 0.8],
        frame_hz: int = 20,
        thresh: float = 0.5,
        class_weights: dict = None,
        loss_type: str = "ce",  # "ce", "bce", "soft_hamming"
        loss_temperature: float = 1.0,
        label_smoothing_temporal: float = 0.0,
    ):
        super().__init__()
        self.frame_hz = frame_hz
        self.bin_times = bin_times
        self.bin_frames = bin_times_to_frames(bin_times, frame_hz)

        self.projection_window_extractor = ProjectionWindow(bin_times, frame_hz, thresh)
        self.codebook = Codebook(self.bin_frames)
        self.requires_grad_(False)
        self.lid_n_classes = 3  # Language ID number of classes (example)

        # Loss configuration
        self.loss_type = loss_type
        self.loss_temperature = loss_temperature
        self.label_smoothing_temporal = label_smoothing_temporal

        # Hamming Distance Matrix (4-bit) for independent speaker soft labels
        # 4 bits = 16 states
        if self.loss_type == "soft_hamming":
            self.register_buffer(
                "hamming_dist_matrix_4bit", self._create_hamming_distance_matrix(4)
            )

        # クラス重みの設定（ラベル15と240に重み0.1を適用）
        self.class_weights = None
        if class_weights is not None:
            weights = torch.ones(self.n_classes)
            for label, weight in class_weights.items():
                if 0 <= label < self.n_classes:
                    weights[label] = weight
            self.class_weights = weights

    def _create_hamming_distance_matrix(self, n_bits: int) -> Tensor:
        """Create a pairwise Hamming distance matrix for n_bits."""
        n_classes = 2**n_bits
        # Create all binary vectors of shape (n_classes, n_bits)
        # Using the same logic as codebook.create_code_vectors just locally
        code_vectors = torch.zeros((n_classes, n_bits))
        for i in range(n_classes):
            b = bin(i).replace("0b", "")
            # zero pad
            b = "0" * (n_bits - len(b)) + b
            for j, val in enumerate(b):
                code_vectors[i, j] = float(val)
        
        # Calculate pairwise hamming distance
        # dist[i, j] = sum(|v[i] - v[j]|)
        # using efficient broadcasting
        # (N, 1, D) - (1, N, D) -> (N, N, D) -> sum -> (N, N)
        dist = torch.abs(code_vectors.unsqueeze(1) - code_vectors.unsqueeze(0)).sum(-1)
        return dist

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(\n"
        s += f" {self.codebook},\n"
        s += f" {self.projection_window_extractor}\n"
        s += f" loss_type={self.loss_type}, temp={self.loss_temperature}, temporal_smooth={self.label_smoothing_temporal}\n"
        s += ")"
        return s

    @property
    def n_classes(self) -> int:
        return self.codebook.n_classes

    @property
    def n_bins(self) -> int:
        return self.codebook.n_bins

    def get_labels(self, va: Tensor) -> Tensor:
        """ボイスアクティビティ（va）からVAPラベル（インデックス）を生成します。"""
        projection_bins = self.projection_window_extractor(va)
        return self.codebook(projection_bins.type(va.dtype))

    def get_soft_targets(self, va: Tensor) -> Tensor:
        """
        Generate decomposed soft targets based on Hamming distance and independent speakers.
        Args:
            va: (B, T, 2)
        Returns:
            soft_targets: (B, T, 256)
        """
        # 1. Get binary projection bins (B, T, 2, 4) -> (B, T, 8) flattened
        # But we need separated by speaker for decomposed soft labels
        projection_bins = self.projection_window_extractor(va) # (B, T, 2, 4)
        
        # Convert bins to indices 0-15 per speaker
        # Speaker 0 bins: (B, T, 4)
        # Speaker 1 bins: (B, T, 4)
        
        def bins_to_idx(bins_tensor):
            # bins_tensor: (..., 4)
            # weights: [8, 4, 2, 1] (binary 3, 2, 1, 0)
            # Our `bins_to_idx` used w=[1, 2, 4, 8]. So it matches the z[0]..z[3] logic (LSB first).
            w = torch.tensor([1, 2, 4, 8], device=bins_tensor.device, dtype=bins_tensor.dtype)
            return (bins_tensor * w).sum(-1).long()

        # Extract indices for each speaker
        # projection_bins: (B, T, 2, 4)
        idx_s0 = bins_to_idx(projection_bins[..., 0, :]) # (B, T)
        idx_s1 = bins_to_idx(projection_bins[..., 1, :]) # (B, T)

        # 2. Compute Soft Distributions per speaker (Independent)
        # dist matrix: (16, 16)
        # Select row for each target index
        # We need (B, T, 16) distribution
        D = self.hamming_dist_matrix_4bit.to(va.device) # (16, 16)
        
        # F.embedding can be used to lookup the row D[idx]
        dist_s0 = F.embedding(idx_s0, D) # (B, T, 16)
        dist_s1 = F.embedding(idx_s1, D) # (B, T, 16)
        
        # Apply Softmax with Temperature
        # Higher temperature -> softer
        # q(i) = exp(-d / T) / sum(exp(-d / T))
        # We use negative distance as "logit"
        temp = self.loss_temperature
        prob_s0 = F.softmax(-dist_s0 / temp, dim=-1) # (B, T, 16)
        prob_s1 = F.softmax(-dist_s1 / temp, dim=-1) # (B, T, 16)

        # 3. Combine via Outer Product using einsum
        # (B, T, i) * (B, T, j) -> (B, T, i, j) -> flatten to (B, T, i*16 + j)
        
        prob_joint = torch.einsum("btj, bti -> btji", prob_s1, prob_s0) # (B, T, 16_s1, 16_s0)
        prob_joint = rearrange(prob_joint, "b t s1 s0 -> b t (s1 s0)") # (B, T, 256)

        # 4. Temporal Smoothing (Optional)
        # Apply Conv1d over T dimension
        if self.label_smoothing_temporal > 0:
            # Kernel design: Simple Gaussian-ish or Triangle
            # e.g. [0.25, 0.5, 0.25] if smoothing is small?
            alpha = 0.25 # decent default
            kernel = torch.tensor([alpha, 1.0 - 2*alpha, alpha], device=va.device).view(1, 1, 3)
            # We want to apply this to each of 256 channels independently over time.
            # shape (B, 256, T)
            p_in = rearrange(prob_joint, "b t c -> b c t")
            # padding=1 to keep same size
            # groups=256 to apply same kernel to each channel independently
            # weight needs to be (256, 1/groups, k) -> (256, 1, 3)
            kernel = kernel.repeat(256, 1, 1)
            
            p_smooth = F.conv1d(p_in, kernel, padding=1, groups=256)
            prob_joint = rearrange(p_smooth, "b c t -> b t c")
            
            # Renormalize just in case (conv might drift sum slightly at edges)
            prob_joint = prob_joint / (prob_joint.sum(-1, keepdim=True) + 1e-6)

        return prob_joint

    def _calculate_ce_loss(
        self,
        logits: Tensor,
        labels: Tensor,
        reduction: str = "mean",
        class_weights: Tensor = None,
    ) -> Tensor:
        """クロスエントロピー損失を計算する共通ヘルパー関数。"""
        if logits.ndim != 3:
            raise ValueError(f"Expected logits shape (B, T, C), got {logits.shape}")
        
        # Check if using Soft Labels (labels dim match logits dim)
        is_soft = (labels.ndim == 3) and (labels.shape[-1] == logits.shape[-1])

        if not is_soft and labels.ndim != 2:
            raise ValueError(f"Expected labels shape (B, T), got {labels.shape}")

        # フレーム数をラベルに合わせる
        n_frames = min(logits.shape[1], labels.shape[1])
        logits = logits[:, :n_frames]
        labels = labels[:, :n_frames]

        if is_soft:
            # Soft Label Loss (KLDiv)
            # KLDiv expects input as log_softmax, target as probability
            log_probs = F.log_softmax(logits, dim=-1)
            
            # batchmode handling
            loss = F.kl_div(log_probs, labels, reduction="none") # (B, T, C)
            loss = loss.sum(-1) # Sum over classes -> (B, T)
            
            if reduction == "mean":
                if class_weights is not None:
                     pass
                loss = loss.mean()
            elif reduction == "sum":
                loss = loss.sum()
            # if none, returns (B, T)
            return loss
            
        else:
            # Hard Label Loss
            loss = F.cross_entropy(
                rearrange(logits, "b n d -> (b n) d"),
                rearrange(labels, "b n -> (b n)"),
                weight=class_weights,
                reduction=reduction,
            )

            if reduction == "none":
                loss = rearrange(loss, "(b n) -> b n", b=logits.shape[0])
            return loss

    def loss_vap(
        self, logits: Tensor, labels: Tensor, reduction: str = "mean"
    ) -> Tensor:
        """VAPタスクのクロスエントロピー損失を計算します。"""
        class_weights = self.class_weights
        if class_weights is not None:
            class_weights = class_weights.to(logits.device)
        return self._calculate_ce_loss(logits, labels, reduction, class_weights)

    def loss_vad(self, vad_logits: Tensor, vad_targets: Tensor) -> Tensor:
        """VADタスクのバイナリクロスエントロピー損失を計算します。"""
        n = vad_logits.shape[-2]
        return F.binary_cross_entropy_with_logits(vad_logits, vad_targets[:, :n])

    def _process_binary_metric(
        self,
        p_event: Tensor,
        events: dict,
        positive_key: str,
        negative_key: str,
        aggregate: bool = False,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """
        バイナリ評価指標（例: shift/hold）の予測とターゲットを抽出するヘルパー関数。

        Args:
            p_event (Tensor): ポジティブイベントが発生する確率。
            events (dict): イベントのタイムスタンプ情報。
            positive_key (str): ポジティブイベントのキー (ターゲット=1)。
            negative_key (str): ネガティブイベントのキー (ターゲット=0)。
            aggregate (bool): Trueの場合、イベント区間全体の平均確率を計算。

        Returns:
            tuple[list[Tensor], list[Tensor]]: (予測リスト, ターゲットリスト)
        """
        preds, targets = [], []
        batch_size = p_event.shape[0]

        def extract(key: str, target_val: float):
            if key not in events:
                return
            for b in range(batch_size):
                for start, end, speaker in events[key][b]:
                    if start >= end:
                        continue
                    prob_slice = p_event[b, start:end, speaker]
                    if aggregate:
                        preds.append(torch.mean(prob_slice).unsqueeze(0))
                        targets.append(
                            torch.full((1,), target_val, device=p_event.device)
                        )
                    else:
                        preds.append(prob_slice)
                        targets.append(torch.full_like(prob_slice, target_val))

        extract(positive_key, 1.0)
        extract(negative_key, 0.0)
        return preds, targets

    @torch.no_grad()
    def extract_prediction_and_targets(
        self, p_now: Tensor, p_fut: Tensor, events: Dict
    ) -> Tuple[Dict[str, Tensor | None], Dict[str, Tensor | None]]:
        """
        モデルの確率出力から評価指標（shift/holdなど）を計算するための
        予測値と正解ターゲットを抽出します。
        """
        metric_definitions = {
            "hs": {"p": p_now, "pos": "shift", "neg": "hold", "agg": True},
            "pred_shift": {
                "p": p_fut,
                "pos": "pred_shift",
                "neg": "pred_shift_neg",
                "agg": True,
            },
            "ls": {"p": p_fut, "pos": "long", "neg": "short", "agg": True},
        }

        all_preds, all_targets = {}, {}
        for metric, conf in metric_definitions.items():
            all_preds[metric], all_targets[metric] = self._process_binary_metric(
                conf["p"], events, conf["pos"], conf["neg"], aggregate=conf["agg"]
            )

        # 全てのリストを結合して一つのテンソルにまとめる
        out_preds = {k: torch.cat(v) if v else None for k, v in all_preds.items()}
        out_targets = {
            k: torch.cat(v).long() if v else None for k, v in all_targets.items()
        }
        return out_preds, out_targets

    def probs_next_speaker_aggregate(
        self,
        probs: Tensor,
        from_bin: int = 0,
        to_bin: int = 3,
        scale_with_bins: bool = False,
    ) -> Tensor:
        assert probs.ndim == 3, (
            f"Expected probs of shape (B, n_frames, n_classes) but got {probs.shape}"
        )
        idx = torch.arange(self.codebook.n_classes).to(probs.device)
        states = self.codebook.decode(idx)

        if scale_with_bins:
            states = states * torch.tensor(self.bin_frames)
        abp = states[:, :, from_bin : to_bin + 1].sum(-1)  # sum speaker activity bins
        # Dot product over all states
        p_all = torch.einsum("bid,dc->bic", probs, abp)
        # normalize
        p_all /= p_all.sum(-1, keepdim=True) + 1e-5
        return p_all

    def get_probs(self, logits: Tensor) -> Dict[str, Tensor]:
        """
        Extracts labels from the voice-activity, va.
        The labels are based on projections of the future and so the valid
        frames with corresponding labels are strictly less then the original number of frams.

        Arguments:
        -----------
        logits:     torch.Tensor (B, N_FRAMES, N_CLASSES)
        va:         torch.Tensor (B, N_FRAMES, 2)

        Return:
        -----------
            Dict[probs, p, p_bc, labels]  which are all torch.Tensors
        """

        assert logits.shape[-1] == self.n_classes, (
            f"Logits have wrong shape. {logits.shape} != (..., {self.n_classes}) that is (B, N_FRAMES, N_CLASSES)"
        )

        probs = logits.softmax(dim=-1)

        return {
            "probs": probs,
            "p_now": self.probs_next_speaker_aggregate(
                probs=probs, from_bin=0, to_bin=1
            ),
            "p_future": self.probs_next_speaker_aggregate(
                probs=probs, from_bin=2, to_bin=3
            ),
            "p_tot": self.probs_next_speaker_aggregate(
                probs=probs, from_bin=0, to_bin=3
            ),
        }


if __name__ == "__main__":
    # モジュールの動作確認
    objective = ObjectiveVAP()
    print(objective)

    # ダミーデータでラベル生成をテスト
    dummy_va = torch.rand(2, 200, 2)  # (batch, frames, speakers)
    labels = objective.get_labels(dummy_va)
    print(f"Dummy VA shape: {dummy_va.shape}")
    print(f"Generated labels shape: {labels.shape}")
    
    # Test soft targets
    obj_soft = ObjectiveVAP(loss_type="soft_hamming", loss_temperature=1.0, label_smoothing_temporal=0.1)
    soft_targets = obj_soft.get_soft_targets(dummy_va)
    print(f"Soft targets shape: {soft_targets.shape}")
    print(f"Soft targets sum check (frame 0): {soft_targets[0, 0].sum()}")
