import random

import torch

import utilities.transforms as VT


class AudioAugmentation:
    """
    音声データにオーグメンテーションを適用するクラス。
    PyTorch LightningのCallbackから独立させました。
    GPU最適化版。
    """

    def __init__(
        self,
        probability: float = 0.5,
        noise_amplitude: float = 0.01,
        pitch_steps: list[int] = [-2, -1, 1, 2],
        freq_mask_param: int = 100,
        iid_masks: bool = True,
        sample_rate: int = 16_000,
        device: str = "cpu",
    ):
        self.device = device
        self._use_gpu = device in ["cuda", "mps"] and device != "cpu"

        # 内部のオーグメンテーションロジック（GPU対応）
        self.augmentation = VT.Augmentation(
            probability=probability,
            noise_amplitude=noise_amplitude,
            pitch_steps=pitch_steps,
            freq_mask_param=freq_mask_param,
            iid_masks=iid_masks,
            sample_rate=sample_rate,
            device=device,
        )

    def __call__(self, batch: dict) -> dict:
        """
        バッチ内の波形データにオーグメンテーションを適用します。
        on_train_batch_startの代わりに、このメソッドを直接呼び出します。
        GPU最適化版。
        """
        if "waveform" in batch and isinstance(batch["waveform"], torch.Tensor):
            # GPU最適化: waveformがすでにGPU上にある場合、そのまま処理
            waveform = batch["waveform"]
            if self._use_gpu and waveform.device.type != self.device:
                # 必要に応じてデバイス移動（通常は既にGPU上にあるはず）
                waveform = waveform.to(self.device)

            batch["waveform"] = self.augmentation(waveform)
        return batch


class SymmetricSpeakers:
    """
    確率的に話者を反転させるクラス。
    VADテンソルのチャンネルを入れ替えます。
    GPU最適化版。
    """

    def __init__(
        self,
        probability: float = 0.5,
        on_train: bool = True,
        on_val: bool = False,
        on_test: bool = False,
        device: str = "cpu",
    ):
        self.probability = probability
        self.on_train = on_train
        self.on_val = on_val
        self.on_test = on_test
        self.device = device

        # GPU使用時は確率判定もGPUで実行
        self._use_gpu = device in ["cuda", "mps"] and device != "cpu"

    def get_flipped_batch(self, batch: dict) -> dict:
        """バッチ内のサンプルを反転させるヘルパー関数（GPU最適化）"""
        # GPU最適化: テンソル操作を一括で実行
        for k, v in batch.items():
            if k == "vad" and isinstance(v, torch.Tensor):
                # VADのチャンネルを反転 (B, N_FRAMES, 2) -> (B, N_FRAMES, 2)
                batch[k] = v.flip(-1)
            elif k == "waveform" and isinstance(v, torch.Tensor):
                if (
                    v.ndim == 3 and v.shape[1] == 2
                ):  # ステレオ音声の場合 (B, 2, N_SAMPLES)
                    # チャンネル次元を反転
                    batch[k] = v.flip(1)
        return batch

    def __call__(self, batch: dict, stage: str) -> dict:
        """
        現在のステージ（'train', 'val', 'test'）に応じて、
        バッチを確率的に反転させます。（GPU最適化版）
        """
        should_flip = False

        if stage == "train" and self.on_train:
            # GPU最適化: 確率判定もGPUテンソルで実行（可能な場合）
            if self._use_gpu:
                # バッチサイズに応じた確率判定をGPUで実行
                batch_size = None
                for v in batch.values():
                    if isinstance(v, torch.Tensor):
                        batch_size = v.shape[0]
                        break

                if batch_size is not None:
                    # バッチ単位でランダム判定（GPU上で実行）
                    rand_tensor = torch.rand(
                        1,
                        device=self.device
                        if hasattr(torch, "cuda") and torch.cuda.is_available()
                        else "cpu",
                    )
                    should_flip = rand_tensor.item() < self.probability
                else:
                    should_flip = random.random() < self.probability
            else:
                should_flip = random.random() < self.probability

        elif stage == "val" and self.on_val:
            # 検証/テスト時は確率なしで、フラグがTrueなら常に反転
            should_flip = True
        elif stage == "test" and self.on_test:
            should_flip = True

        if should_flip:
            return self.get_flipped_batch(batch)

        return batch


class GPUOptimizedCallbacks:
    """
    GPU最適化されたコールバックの統合クラス
    複数のコールバックを効率的にバッチ処理
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._use_gpu = device in ["cuda", "mps"] and device != "cpu"
        self.callbacks = []

    def add_callback(self, callback):
        """コールバックを追加"""
        self.callbacks.append(callback)

    def __call__(self, batch: dict, stage: str = "train") -> dict:
        """すべてのコールバックを効率的に実行"""
        # GPU最適化: すべてのテンソル操作を一括実行
        for callback in self.callbacks:
            if hasattr(callback, "__call__"):
                if hasattr(callback, "device"):
                    # デバイス情報を持つコールバックの場合
                    callback.device = self.device

                # ステージ情報が必要なコールバックの処理
                if "stage" in callback.__call__.__code__.co_varnames:
                    batch = callback(batch, stage)
                else:
                    batch = callback(batch)

        return batch


if __name__ == "__main__":
    import torch

    # 16000Hz音声として0.5秒分のサンプルデータを生成
    sample_rate = 16000
    duration = 0.5  # 0.5秒
    n_samples = int(sample_rate * duration)  # 8000サンプル
    frame_time = 0.02  # 20ms フレーム
    n_frames = int(duration / frame_time)  # 25フレーム

    batch = {
        "waveform": torch.stack(
            [
                # ステレオ音声 (B=2, C=2, N=8000)
                torch.randn(2, n_samples),  # バッチ1: ランダム音声データ
                torch.randn(2, n_samples),  # バッチ2: ランダム音声データ
            ]
        ),
        "vad": torch.stack(
            [
                # VAD (B=2, N_FRAMES=25, 2) - 20ms フレームで0.5秒分
                torch.rand(n_frames, 2),  # バッチ1: ランダムVADデータ
                torch.rand(n_frames, 2),  # バッチ2: ランダムVADデータ
            ]
        ),
        "other_info": ["sample1", "sample2"],  # その他の情報
    }

    print("=== 元のバッチ ===")
    print(batch)

    augmenter = AudioAugmentation(probability=1.0)  # 常にオーグメンテーションを適用
    augmented_batch = augmenter(batch)
    print("\n=== オーグメンテーション後のバッチ ===")
    print(augmented_batch)

    sym_speakers = SymmetricSpeakers(probability=1.0, on_train=True)
    flipped_batch = sym_speakers(batch, stage="train")
    print("\n=== 話者反転後のバッチ ===")
    print(flipped_batch)
