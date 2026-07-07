from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from config import VapConfig
from models.encoder import EncoderCPC
from models.gpt import GPT, GPTStereo, Combinator, TextCrossAttentionLayer
from models.text_encoder import TextEncoder
from utilities.utils import vad_fill_silences, vad_omit_spikes


class VapGPT(nn.Module):
    """音声・テキスト融合によるVAP（Voice Activity Projection）モデル。

    フレーム単位の音声特徴量とテキストトークン列を入力として、
    VAD・将来発話割合・次単語予測を出力する。
    """

    def __init__(self, conf: Optional[VapConfig] = None):
        super().__init__()
        if conf is None:
            conf = VapConfig()
        self.conf = conf
        self.sample_rate = conf.sample_rate
        self.frame_hz = conf.frame_hz

        # --- Audio Encoder (CPC) ---
        self.audio_encoder = EncoderCPC(
            cpc_model_pt=conf.cpc_model_pt,
            load_pretrained=bool(conf.load_pretrained),
            freeze=conf.freeze_encoder,
            lim_context_sec=conf.context_limit_cpc_sec,
            frame_hz=conf.frame_hz,
        )

        # --- チャネル処理方式 ---
        self.use_stereo = conf.use_stereo
        if self.use_stereo:
            # Stereo: チャネル分離処理（GPT per channel + GPTStereo cross-attention）
            self.ar_channel = GPT(
                dim=conf.dim,
                dff_k=conf.transformer_dff_k,
                num_layers=conf.num_channel_layers,
                num_heads=conf.num_heads,
                dropout=conf.transformer_dropout,
            )
            self.ar_stereo = GPTStereo(
                dim=conf.dim,
                dff_k=conf.transformer_dff_k,
                num_layers=conf.num_cross_layers,
                num_heads=conf.num_heads,
                dropout=conf.transformer_dropout,
            )
        else:
            # 結合: 早期 concat → 単一 GPT
            self.audio_proj = nn.Linear(conf.dim * 2, conf.dim)

        # --- Text Encoder + Cross-Attention ---
        self.use_text = conf.use_text
        if self.use_text:
            self.text_encoder = TextEncoder(
                vocab_size=conf.vocab_size,
                dim=conf.dim,
                glove_dim=conf.glove_dim,
                glove_path=conf.glove_path,
                vocab_path=conf.vocab_path,
                dropout=conf.text_dropout,
            )
            self.text_cross_attn = nn.ModuleList([
                TextCrossAttentionLayer(
                    dim=conf.dim,
                    num_heads=conf.num_heads,
                    dropout=conf.transformer_dropout,
                )
                for _ in range(conf.num_text_cross_layers)
            ])
        else:
            self.text_encoder = None
            self.text_cross_attn = None

        if not self.use_stereo:
            # 結合モード: Fusion + 単一 GPT
            self.fusion = nn.Sequential(
                nn.Linear(conf.dim, conf.dim),
                nn.ReLU(),
                nn.Dropout(conf.fusion_dropout),
            )
            self.transformer = GPT(
                dim=conf.dim,
                dff_k=conf.transformer_dff_k,
                num_layers=conf.num_transformer_layers,
                num_heads=conf.num_heads,
                dropout=conf.transformer_dropout,
            )

        # --- Output Heads ---
        self.va_head = nn.Linear(conf.dim, 2, bias=False)
        self.onset_proximity_head = nn.Linear(conf.dim, 2, bias=False)
        self.filler_dropout = nn.Dropout(0.3)
        self.filler_head = nn.Linear(conf.dim, conf.num_filler_classes, bias=False)

        # エンコーダを凍結する
        if conf.freeze_encoder:
            logger.info("CPCエンコーダの重みを凍結")
            self.audio_encoder.freeze()

    def _encode_audio_cpc(
        self, audio: Tensor, n_frames: int
    ) -> Tensor:
        """CPCエンコーダで音声特徴量を抽出する。

        Args:
            audio: (B, 2, total_samples) ステレオ音声
            n_frames: ウィンドウ内のフレーム数

        Returns:
            (B, n_frames, dim) 音声特徴量
        """
        B = audio.shape[0]

        x1 = self.audio_encoder(audio[:, :1])  # (B, T', dim)
        x2 = self.audio_encoder(audio[:, 1:])  # (B, T', dim)
        t = min(x1.shape[1], n_frames)
        audio_feat = self.audio_proj(
            torch.cat([x1[:, :t], x2[:, :t]], dim=-1)
        )  # (B, t, dim)

        # n_framesに満たない場合はゼロ埋めする
        if t < n_frames:
            pad = torch.zeros(
                B, n_frames - t, audio_feat.shape[-1], device=audio.device
            )
            audio_feat = torch.cat([audio_feat, pad], dim=1)

        return audio_feat

    def _gather_text_for_frames(
        self, text_hidden: Tensor, text_token_positions: Tensor, n_frames: int
    ) -> Tensor:
        """各音声フレームに対応するテキスト隠れ状態を収集する。

        フレームtにおけるテキストコンテキストは、
        text_token_positions[b, i] <= t を満たす最後のトークンiの隠れ状態。
        該当トークンが無いフレームはゼロベクトルになる。

        Args:
            text_hidden: (B, N, dim) テキストエンコーダの出力
            text_token_positions: (B, N) 各トークンのフレーム位置（ウィンドウ相対、-1はパディング）
            n_frames: フレーム数 T

        Returns:
            (B, T, dim) フレームごとのテキスト特徴量
        """
        B, N, dim = text_hidden.shape
        device = text_hidden.device

        # フレームインデックス: (1, T, 1)
        frame_indices = torch.arange(n_frames, device=device).unsqueeze(0).unsqueeze(-1)
        # トークン位置: (B, 1, N)
        positions = text_token_positions.unsqueeze(1)

        # フレームtでトークンiが見える条件: position <= t かつ position >= 0
        visible = (positions <= frame_indices) & (positions >= 0)  # (B, T, N)

        # 各(b, t)で最後の可視トークンのインデックスを求める
        visible_positions = positions.expand(B, n_frames, N).clone()
        visible_positions[~visible] = -1
        last_idx = visible_positions.argmax(dim=-1)  # (B, T)

        # テキストコンテキストが無いフレームを検出する
        has_context = visible.any(dim=-1)  # (B, T)

        # テキスト隠れ状態をgatherで収集する
        last_idx_expanded = last_idx.unsqueeze(-1).expand(B, n_frames, dim)
        text_feat = torch.gather(text_hidden, 1, last_idx_expanded)  # (B, T, dim)

        # テキストコンテキストが無いフレームをゼロにする
        text_feat[~has_context] = 0.0

        return text_feat

    def vad_loss(self, vad_output: Tensor, vad: Tensor) -> Tensor:
        """VAD出力に対するBCE with logits損失を計算する。

        Args:
            vad_output: (B, T, 2) モデルのVAD logits出力
            vad: (B, T, 2) 正解VADラベル

        Returns:
            スカラー損失値
        """
        return F.binary_cross_entropy_with_logits(vad_output, vad)

    @torch.no_grad()
    def vad(
        self,
        audio: Tensor,
        text_tokens: Tensor,
        text_token_positions: Tensor,
        n_frames: int,
        max_fill_silence_time: float = 0.02,
        max_omit_spike_time: float = 0.02,
        vad_cutoff: float = 0.5,
    ) -> Tensor:
        """モデル出力からバイナリVADを抽出する。

        短い無音区間の補完とスパイク除去の後処理を行う。

        Args:
            audio: (B, 2, total_samples)
            text_tokens: (B, max_text_tokens)
            text_token_positions: (B, max_text_tokens)
            n_frames: フレーム数
            max_fill_silence_time: 補完する最大無音時間（秒）
            max_omit_spike_time: 除去する最大スパイク時間（秒）
            vad_cutoff: VAD判定閾値

        Returns:
            (B, T, 2) バイナリVAD
        """
        output = self(audio, text_tokens, text_token_positions, n_frames)
        vad_result = (output["vad"].sigmoid() >= vad_cutoff).float()
        for b in range(vad_result.shape[0]):
            vad_result[b] = vad_fill_silences(
                vad_result[b],
                max_fill_time=max_fill_silence_time,
                frame_hz=self.frame_hz,
            )
            vad_result[b] = vad_omit_spikes(
                vad_result[b],
                max_omit_time=max_omit_spike_time,
                frame_hz=self.frame_hz,
            )
        return vad_result

    def forward(
        self,
        audio: Tensor,
        text_tokens: Tensor,
        text_token_positions: Tensor,
        n_frames: int,
        cpc_feat_1: Optional[Tensor] = None,
        cpc_feat_2: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """フォワードパス。

        Args:
            audio: (B, 2, total_samples) ステレオ音声
            text_tokens: (B, max_text_tokens) パディング済みトークンID
            text_token_positions: (B, max_text_tokens) 各トークンのフレーム位置（-1はパディング）
            n_frames: ウィンドウ内のフレーム数（例: 400）
            cpc_feat_1: (B, T, 256) 事前計算済みCPC特徴量（話者1）。Noneならaudioから計算
            cpc_feat_2: (B, T, 256) 事前計算済みCPC特徴量（話者2）

        Returns:
            dict: vad (B, T, 2), onset_proximity (B, T, 2),
                  filler_logits (B, T, num_filler_classes)
        """
        # 1. 音声特徴量の抽出
        if cpc_feat_1 is not None and cpc_feat_2 is not None:
            # 事前計算済み特徴量を使用（100Hz → 20Hz）
            x1 = self.audio_encoder.downsample(cpc_feat_1)
            x2 = self.audio_encoder.downsample(cpc_feat_2)
            t = min(x1.shape[1], n_frames)
            x1 = x1[:, :t]
            x2 = x2[:, :t]
            if t < n_frames:
                B = x1.shape[0]
                pad = torch.zeros(B, n_frames - t, x1.shape[-1], device=x1.device)
                x1 = torch.cat([x1, pad], dim=1)
                x2 = torch.cat([x2, pad], dim=1)
        else:
            B = audio.shape[0]
            x1 = self.audio_encoder(audio[:, :1])
            x2 = self.audio_encoder(audio[:, 1:])
            t = min(x1.shape[1], n_frames)
            x1 = x1[:, :t]
            x2 = x2[:, :t]
            if t < n_frames:
                pad = torch.zeros(B, n_frames - t, x1.shape[-1], device=audio.device)
                x1 = torch.cat([x1, pad], dim=1)
                x2 = torch.cat([x2, pad], dim=1)

        # 2. Transformer
        if self.use_stereo:
            # Stereo: チャネル別 Self-Attention → Cross-Attention → Combinator
            o1 = self.ar_channel(x1)["x"]
            o2 = self.ar_channel(x2)["x"]
            stereo_out = self.ar_stereo(o1, o2)
            fused = stereo_out["x"]  # combined output
        else:
            # 結合: concat → proj → Fusion → 単一 GPT
            audio_feat = self.audio_proj(torch.cat([x1, x2], dim=-1))
            fused = self.fusion(audio_feat)
            fused = self.transformer(fused)["x"]

        # 4. テキスト Cross-Attention（use_text=True の場合のみ）
        if self.use_text and self.text_cross_attn is not None:
            text_hidden = self.text_encoder(text_tokens)  # (B, N, dim)
            # パディングマスク: position == -1 のトークンを無視
            text_pad_mask = (text_token_positions == -1)  # (B, N), True=無視

            # causal テキストマスク: 各フレーム t は position <= t の単語のみ参照。
            # これによりオンライン時に未来単語（次話者 onset 後の語など）を
            # 参照するリークを防ぐ。text_causal=False なら従来の非因果動作。
            attn_mask = None
            if getattr(self.conf, "text_causal", True):
                B_, N_ = text_token_positions.shape
                T_ = fused.shape[1]
                frame_idx = torch.arange(T_, device=fused.device).view(1, T_, 1)  # (1,T,1)
                pos = text_token_positions.view(B_, 1, N_)                          # (B,1,N)
                visible = (pos != -1) & (pos <= frame_idx)                          # (B,T,N)
                attn_mask = torch.zeros(
                    B_, T_, N_, device=fused.device, dtype=fused.dtype
                )
                attn_mask = attn_mask.masked_fill(~visible, torch.finfo(fused.dtype).min)
                # attn_mask がパディングも -inf にしているため key_padding_mask は不要
                text_pad_mask = None

            for cross_layer in self.text_cross_attn:
                fused = cross_layer(fused, text_hidden, text_pad_mask, attn_mask)

        # 6. 各ヘッドで予測する
        vad = self.va_head(fused)
        onset_proximity = torch.sigmoid(self.onset_proximity_head(fused))
        filler_logits = self.filler_head(self.filler_dropout(fused))

        return {
            "vad": vad,
            "onset_proximity": onset_proximity,
            "filler_logits": filler_logits,
        }


if __name__ == "__main__":
    import time

    logger.info("VapGPTモデルのテストを開始")
    conf = VapConfig()
    model = VapGPT(conf)
    logger.info(f"モデル構造:\n{model}")

    # 学習可能パラメータの確認
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"総パラメータ数: {total_params:,}")
    logger.info(f"学習可能パラメータ数: {trainable_params:,}")

    # デバイス設定
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    logger.info(f"使用デバイス: {device}")
    model.to(device)

    # ダミー入力の作成
    batch_size = 4
    duration_sec = 20
    n_frames = duration_sec * conf.frame_hz  # 400
    n_samples = conf.sample_rate * duration_sec
    dummy_audio = torch.randn(batch_size, 2, n_samples).to(device)
    dummy_text_tokens = torch.randint(
        0, conf.vocab_size, (batch_size, conf.max_text_tokens)  # text tokens use full vocab
    ).to(device)
    # テキストトークン位置: 最初の30トークンに有効な位置を設定、残りは-1
    dummy_positions = torch.full(
        (batch_size, conf.max_text_tokens), -1, dtype=torch.long
    ).to(device)
    for b in range(batch_size):
        n_valid = 30
        dummy_positions[b, :n_valid] = torch.sort(
            torch.randint(0, n_frames, (n_valid,))
        )[0]

    # フォワードパスの実行
    with torch.no_grad():
        s = time.time()
        output = model(dummy_audio, dummy_text_tokens, dummy_positions, n_frames)
        elapsed = time.time() - s
        logger.info(f"フォワードパス実行時間: {elapsed:.4f}秒")

    logger.info(f"出力キー: {list(output.keys())}")
    for key, val in output.items():
        logger.info(f"  {key}: shape={val.shape}")
