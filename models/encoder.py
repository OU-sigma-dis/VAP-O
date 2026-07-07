import os

import einops
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from models.encoder_components import get_cnn_layer, load_CPC


class EncoderCPC(nn.Module):
    """
    Encoder: waveform -> h
    pretrained: default='cpc'

    A simpler version of the Encoder
    check paper (branch) version to see other encoders...
    """

    def __init__(
        self,
        cpc_model_pt="",
        load_pretrained=True,
        freeze=True,
        lim_context_sec: float = -1,
        frame_hz: int = 20,
    ):
        super().__init__()
        self.sample_rate = 16000
        self.encoder = load_CPC(cpc_model_pt, load_pretrained)
        self.output_dim = self.encoder.gEncoder.conv4.out_channels
        self.dim = self.output_dim
        self.frame_hz = frame_hz
        self.hop_length = int(self.sample_rate / self.frame_hz)  # 16000 / 50 = 320
        kernel_size = int((16000 / 50 / 160) * 2.5)
        stride = int(self.sample_rate / self.frame_hz / 160)

        kernel_size = int(100 / self.frame_hz)
        stride = int(100 / self.frame_hz)

        self.downsample = get_cnn_layer(
            dim=self.output_dim,
            kernel=[kernel_size],
            stride=[stride],
            dilation=[1],
            activation="GELU",
        )

        # 入力の長さを制限
        self.lim_context_sec = lim_context_sec

        # 1000が割り切れるようにする
        self.STEP_SIZE_BY_CONTEXT_LIM = {
            15: 10,
            10: 10,
            5: 25,
            3: 50,
            2: 50,
            1: 100,
        }

        if freeze:
            self.freeze()
        else:
            # CNNs are frozen by default
            self.unfreeze()

    def get_default_conf(self):
        return {""}

    def freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        logger.info(f"Froze {self.__class__.__name__}!")

    def unfreeze(self):
        # Unfreeze only the autoregressive part (except the CNN layers)
        self.freeze()
        self.encoder.gAR.requires_grad_(True)
        logger.info(f"Trainable {self.__class__.__name__}!")

    def forward(self, waveform, only_feature_extractor: int = 0):
        if waveform.ndim < 3:
            waveform = waveform.unsqueeze(1)  # channel dim

        # Backwards using only the encoder encounters:
        # ---------------------------------------------------
        # RuntimeError: one of the variables needed for gradient computation
        # has been modified by an inplace operation:
        # [torch.FloatTensor [4, 256, 1000]], which is output 0 of ReluBackward0, is at version 1;
        # expected version 0 instead. Hint: enable anomaly detection to find
        # the operation that failed to compute its gradient, with
        # torch.autograd.set_detect_anomaly(True).
        # HOWEVER, if we feed through encoder.gAR we do not encounter that problem...
        if self.lim_context_sec < 0:
            z = self.encoder.gEncoder(waveform)
            z = einops.rearrange(z, "b c n -> b n c")
            z = self.encoder.gAR(z)
            z = self.downsample(z)

        # 入力の長さを制限（処理時間がかかる）
        if self.lim_context_sec > 0:
            FRAME_PER_FEATURE = 320
            DIM_FEATURE = 256
            num_feature = int(waveform.shape[2] / FRAME_PER_FEATURE)  # 20sec -> 1000
            lim_context_n = int(self.lim_context_sec * self.sample_rate)
            z = np.zeros((waveform.shape[0], num_feature, DIM_FEATURE))

            # キャッシュにデータがあるか確認
            cached_idx = []
            for b in range(waveform.shape[0]):
                z_hash = self.hash_tensor(waveform[b, :, :])
                tensor_path = "temp/%dsec/z%s.pt" % (self.lim_context_sec, z_hash)

                if os.path.exists(tensor_path):
                    try:
                        z_ = torch.load(tensor_path)
                    except Exception:
                        continue

                    z[b, :, :] = z_.cpu().clone().detach()
                    del z_
                    cached_idx.append(b)

            batch_size = waveform.shape[0]

            if len(cached_idx) < batch_size:
                logger.info(
                    "z is not cached\t(num_cached={}/{})".format(
                        len(cached_idx), waveform.shape[0]
                    )
                )
            else:
                logger.info(
                    "z is cached\t(num_cached={}/{})".format(
                        len(cached_idx), waveform.shape[0]
                    )
                )

            if len(cached_idx) < batch_size:
                step_size = self.STEP_SIZE_BY_CONTEXT_LIM[self.lim_context_sec]
                for i in range(0, num_feature, step_size):
                    waveform_ = None

                    for j in range(step_size):
                        start_idx = max(
                            (i + j + 1) * FRAME_PER_FEATURE - lim_context_n, 0
                        )
                        end_idx = min(
                            (i + j + 1) * FRAME_PER_FEATURE, waveform.shape[2]
                        )

                        for b in range(batch_size):
                            w_ = waveform[b, :, start_idx:end_idx].clone().detach()

                            # padding
                            if w_.shape[1] < lim_context_n:
                                w_ = torch.cat(
                                    (
                                        torch.zeros(
                                            (w_.shape[0], lim_context_n - w_.shape[1]),
                                            device=w_.device,
                                        ),
                                        w_,
                                    ),
                                    dim=1,
                                )

                            if waveform_ is None:
                                waveform_ = w_
                            else:
                                waveform_ = torch.cat((waveform_, w_), dim=0)

                    waveform_ = waveform_.unsqueeze(1)  # channel dim

                    with torch.no_grad():
                        z_ = self.encoder.gEncoder(waveform_)
                        z_ = einops.rearrange(z_, "b c n -> b n c")
                        z_ = self.encoder.gAR(z_)
                        z_ = self.downsample(z_)

                    batch_size = waveform.shape[0]
                    idx_copied = 0
                    for j in range(step_size):
                        for b in range(batch_size):
                            if b in cached_idx:
                                continue

                            z[b, i + j, :] = (
                                z_[idx_copied, -1, :].to("cpu").detach().numpy().copy()
                            )
                            # print(idx_copied)
                            idx_copied += 1

                    del z_, waveform_, w_

            z = torch.from_numpy(z.astype(np.float32))
            # デバイスは入力テンソルと同じデバイスを使用
            z = z.to(waveform.device)

            # batchごとにzを保存
            for b in range(waveform.shape[0]):
                z_hash = self.hash_tensor(waveform[b, :, :])
                tensor_path = "temp/%dsec/z%s.pt" % (self.lim_context_sec, z_hash)

                # Hash値をファイル名としてzのデータを保存
                if not os.path.exists(tensor_path):
                    os.makedirs(os.path.dirname(tensor_path), exist_ok=True)
                    torch.save(z[b, :, :], tensor_path)

        return z

    def hash_tensor(self, tensor):
        return hash(tuple(tensor.reshape(-1).tolist()))


def test():
    print("Testing EncoderCPC...")

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # モデル初期化
    try:
        encoder = EncoderCPC(
            cpc_model_pt="",  # CPCモデルのパスが必要な場合は適切に設定してください
            load_pretrained=True,
            freeze=True,
            lim_context_sec=5,  # 5秒に制限
            frame_hz=20,
        ).to(device)
        print("Model initialized successfully")
        print(f"Output dimension: {encoder.output_dim}")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    # ダミー音声データを作成（16kHz、2秒）
    sample_rate = 16000
    duration = 2.0  # seconds
    batch_size = 2
    waveform_length = int(sample_rate * duration)

    # ランダムな音声データを生成
    waveform = torch.randn(batch_size, waveform_length).to(device)
    print(f"Input waveform shape: {waveform.shape}")

    # フォワードパス実行
    try:
        with torch.no_grad():
            features = encoder(waveform)
        print(f"Output features shape: {features.shape}")
        print(f"Output features dtype: {features.dtype}")
        print(f"Output features device: {features.device}")
        print("Forward pass completed successfully!")

    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test()
