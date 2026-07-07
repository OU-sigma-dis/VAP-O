import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange



def ffn_block(
    din: int,
    dff: int,
    activation: str = "GELU",
    dropout: float = 0.0,
    bias: bool = False,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(din, dff, bias=bias),
        getattr(nn, activation)(),
        nn.Dropout(p=dropout),
        nn.Linear(dff, din, bias=bias),
    )


class MultiHeadAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float, bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim

        # key, query, value projections for all heads
        self.key = nn.Linear(dim, dim, bias=bias)
        self.query = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)

        # head re-shapers
        self.unstack_heads = Rearrange("b t (h d) -> b h t d", h=self.num_heads)
        self.stack_heads = Rearrange("b h t d -> b t (h d)")

        # regularization
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # output projection
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.scale = 1.0 / math.sqrt(dim)

    def get_scores(self, q: torch.Tensor, k: torch.Tensor):
        """
        Arguments:
            q: (B, heads, T, D)
            k: (B, heads, T, D)

        Return:
            QK:     (B, heads, T, T)
        """
        return torch.einsum("bhid,bhjd->bhij", q, k)

    @staticmethod
    def prepare_causal_mask(T, device, dtype=torch.float32):
        mask = torch.tril(torch.ones((T, T), device=device, dtype=dtype)).view(
            1, 1, T, T
        )
        mask.requires_grad_(False)
        return mask

    def mask_scores(self, qk: torch.Tensor, mask=None):
        T = qk.size(-1)
        if mask is None:
            mask = MultiHeadAttention.prepare_causal_mask(
                T, qk.device, dtype=qk.dtype
            )
        qk = qk.masked_fill(mask == 0, float("-inf"))
        return qk

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, D = Q.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.unstack_heads(self.key(K))  # (B, heads, T, D_head)
        q = self.unstack_heads(self.query(Q))  # (B, heads, T, D_head)
        v = self.unstack_heads(self.value(V))  # (B, heads, T, D_head)

        # QK
        att = self.get_scores(q, k) * self.scale  #  (B, nh, T, T)
        att = self.mask_scores(att, mask)
        att = F.softmax(att, dim=-1)

        # Softmax, dropout, values
        y = self.attn_drop(att) @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # re-assemble all head outputs side by side
        y = self.stack_heads(y)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class MultiHeadAttentionAlibi(MultiHeadAttention):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        bias: bool = False,
        context_limit: int = -1,
    ):
        super().__init__(dim, num_heads, dropout, bias)
        # self.m = torch.tensor(MultiHeadAttentionAlibi.get_slopes(num_heads))
        self.register_parameter(
            "m",
            nn.Parameter(torch.tensor(MultiHeadAttentionAlibi.get_slopes(num_heads))),
        )
        self.m.requires_grad_(False)
        self.mask = None
        self.context_limit = context_limit

    @staticmethod
    def get_slopes(n):
        """
        * aLiBi slopes for heads.
        * m in Figure 3.
        * Source:
            - https://github.com/ofirpress/attention_with_linear_biases/blob/5b327adc6d131e28b40ba58906b30bb469483519/fairseq/models/transformer.py#L742

        Comments:

        In the paper, we only train models that have 2^a heads for some a. This function has
        some good properties that only occur when the input is a power of 2.
        To maintain that even closest_power_of_2 = 2**math.floor(math.log2(n))
        when the number of heads is not a power of 2, we use this workaround.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        # In the paper, we only train models that have 2^a heads for some a. This function has
        # some good properties that only occur when the input is a power of 2. To maintain that even
        # when the number of heads is not a power of 2, we use this workaround.
        if math.log2(n).is_integer():
            slopes = get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            slopes = (
                get_slopes_power_of_2(closest_power_of_2)
                + MultiHeadAttentionAlibi.get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )
        return slopes

    @staticmethod
    def get_relative_bias_matrix(n, num_heads, device, dtype=torch.float32):
        """Relative Bias matrix for aLiBi embeddings"""
        return (
            torch.arange(n, device=device, dtype=dtype)
            .view(1, 1, -1)
            .expand(1, num_heads, -1)
        )

    def get_alibi_mask(self, T: int, device: torch.device, dtype=torch.float32):
        rel_bias_mat = MultiHeadAttentionAlibi.get_relative_bias_matrix(
            T, self.num_heads, device, dtype=dtype
        )
        alibi = rel_bias_mat * self.m.unsqueeze(0).unsqueeze(-1).to(device)

        # Causal mask (standard GPT pask)
        # lower triangle = 1
        # upper triangle = 0
        mask = MultiHeadAttention.prepare_causal_mask(
            T, device, dtype=dtype
        )  # (1, 1, T, T)
        # Repeat to get a mask for each head
        mask = mask.repeat(1, self.num_heads, 1, 1)  # (1, num_heads, T, T)
        # fill "future" information with negative infinity
        mask.masked_fill_(mask == 0, float("-inf"))

        # Add causality mask to alibi  (1, num_heads, T, T)
        alibi = alibi.unsqueeze(-2) + mask
        alibi.requires_grad_(False)  # this should not be trained
        return alibi

    def mask_scores(self, qk: torch.Tensor, mask=None):
        T = qk.size(-1)
        if mask is None:
            if self.mask is None or self.mask.shape[-1] < T:
                mask = self.get_alibi_mask(T, qk.device, dtype=qk.dtype)
                if self.context_limit > 0:
                    for j in range(mask.shape[2]):
                        del_mask_start = 0
                        del_mask_end = max(0, j - self.context_limit + 1)
                        for n in range(del_mask_start, del_mask_end):
                            mask[..., j, n] = float("-inf")

                self.mask = mask
            else:
                mask = self.mask[..., :T, :T]
            # print(mask)
            # print("mask: ", tuple(mask.shape))

        # add aLiBi-mask to qk (see Figure 3.)
        # Addition/translation does not effect softmax (over each row)
        # mentioned in the original representation
        qk = qk + mask.to(qk.device)
        return qk


class TransformerLayer(nn.Module):
    """
    Transformer Layer

    Using pre-layer-normalization: https://arxiv.org/pdf/2002.04745.pdf
    Inspiration: https://nn.labml.ai/transformers/models.html
    AliBI Attention: https://ofir.io/train_short_test_long.pdf
    """

    def __init__(
        self,
        dim: int = 256,
        ffn_dim: int = 768,
        num_heads: int = 4,
        ffn_activation: str = "GELU",
        dropout: float = 0.1,
        cross_attention: bool = False,
        context_limit: int = -1,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.cross_attention = cross_attention

        self.dropout = nn.Dropout(p=dropout)
        self.ln_self_attn = nn.LayerNorm(dim)
        self.ln_ffnetwork = nn.LayerNorm(dim)
        self.mha = MultiHeadAttentionAlibi(
            dim=dim, num_heads=num_heads, dropout=dropout, context_limit=context_limit
        )
        self.ffnetwork = ffn_block(
            dim, ffn_dim, activation=ffn_activation, dropout=dropout
        )

        if cross_attention:
            self.ln_cross_attn = nn.LayerNorm(dim)
            self.mha_cross = MultiHeadAttentionAlibi(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                context_limit=context_limit,
            )

    def forward(self, x, y=None, mask=None):
        """
        Pre-normalization transformer layer

        Arguments:
            x:      (B, T, D)
            y:      (B, T, D)
        Returns:
            out:    (B, T, D)
            attn:   (B, n_heads, T, T)
        """
        # Self attention
        x_norm = self.ln_self_attn(x)
        attn, self_attn = self.mha(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout(attn)

        # cross attention
        if self.cross_attention:
            assert y is not None
            x_norm = self.ln_cross_attn(x)
            cross_attn, cross_attn_weights = self.mha_cross(x_norm, y, y, mask)
            x = x + self.dropout(cross_attn)
        else:
            cross_attn = None
            cross_attn_weights = None

        # Feed forward
        x_norm = self.ln_ffnetwork(x)
        ffn = self.ffnetwork(x_norm)
        x = x + self.dropout(ffn)

        ret = {"x": x, "attn": self_attn}
        if cross_attn is not None:
            ret["cross_attn"] = cross_attn_weights
        return ret


class TransformerStereoLayer(nn.Module):
    """
    Transformer Stereo Layer that combines self-attention, feed-forward, and cross-attention
    """

    def __init__(
        self,
        dim: int = 256,
        ffn_dim: int = 768,
        num_heads: int = 4,
        ffn_activation: str = "GELU",
        dropout: float = 0.1,
        context_limit: int = -1,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout_p = dropout

        self.dropout = nn.Dropout(p=dropout)
        self.ln_self_attn = nn.LayerNorm(dim)
        self.ln_ffnetwork = nn.LayerNorm(dim)
        
        self.mha = MultiHeadAttentionAlibi(
            dim=dim, num_heads=num_heads, dropout=dropout, context_limit=context_limit
        )
        self.ffnetwork = ffn_block(
            dim, ffn_dim, activation=ffn_activation, dropout=dropout
        )
        self.ln_src_attn = nn.LayerNorm(dim)
        self.mha_cross = MultiHeadAttentionAlibi(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            context_limit=context_limit,
        )

    def forward(self, x, y, mask=None):
        """
        Pre-normalization transformer stereo layer

        Arguments:
            x:      (B, T, D) - target sequence
            y:      (B, T, D) - source sequence
        Returns:
            dict with x, attn, cross_attn
        """
        # Self attention
        x_norm = self.ln_self_attn(x)
        attn, self_attn = self.mha(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout(attn)

        # Feed forward
        x_norm = self.ln_ffnetwork(x)
        ffn = self.ffnetwork(x_norm)
        x = x + self.dropout(ffn)

        # Cross attention
        x_norm = self.ln_src_attn(x)
        cross_attn, cross_attn_weights = self.mha_cross(x_norm, y, y, mask)
        x = x + self.dropout(cross_attn)

        return {"x": x, "attn": self_attn, "cross_attn": cross_attn_weights}


class Combinator(nn.Module):
    """
    Combinator module for combining stereo channels
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.h0_a = nn.Linear(dim, dim, bias=False)
        self.h0_b = nn.Linear(dim, dim, bias=False)
        self.ln = nn.LayerNorm(dim)
        self.activation = nn.GELU()

    def forward(self, x1, x2):
        """
        Combine two stereo channels

        Arguments:
            x1: (B, T, D) - first channel
            x2: (B, T, D) - second channel
        Returns:
            combined: (B, T, D) - combined output
        """
        h_a = self.h0_a(x1)
        h_b = self.h0_b(x2)
        combined = self.ln(h_a + h_b)
        return self.activation(combined)


class GPT(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        dff_k: int = 3,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        context_limit: int = -1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    dim=dim,
                    ffn_dim=dff_k * dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    context_limit=context_limit,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, attention=False):
        attention_weights = []

        for layer in self.layers:
            out = layer(x)
            x = out["x"]
            if attention:
                attention_weights.append(out["attn"])

        ret = {"x": x}
        if attention:
            ret["attn"] = torch.stack(attention_weights, dim=1)
        return ret


class TextCrossAttentionLayer(nn.Module):
    """音声フレーム(Q) がテキストトークン(K,V) を参照する Cross-Attention 層。

    ALiBi なし（テキスト位置は音声フレームと異なる構造のため）。
    時間マスク（attn_mask）を渡すと、各フレームが参照できる単語を
    因果的に制限できる（オンライン整合）。渡さなければ従来どおり
    ウィンドウ内の全テキストを参照する（非因果）。
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 3, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 3, dim, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, text_feat: torch.Tensor,
                text_pad_mask: torch.Tensor = None, attn_mask: torch.Tensor = None):
        """
        Args:
            x: (B, T, dim) 音声特徴量
            text_feat: (B, N, dim) テキスト特徴量
            text_pad_mask: (B, N) パディングマスク（True=無視）
            attn_mask: (B, T, N) 加算マスク（-inf で参照禁止）。causal 制限に用いる。
                       None なら時間制約なし（従来の非因果動作）。
        Returns:
            (B, T, dim)
        """
        x_norm = self.ln_q(x)
        kv_norm = self.ln_kv(text_feat)
        am = None
        if attn_mask is not None:
            # (B, T, N) -> (B*num_heads, T, N) に展開して MHA へ渡す
            b, t, n = attn_mask.shape
            am = (attn_mask.unsqueeze(1)
                  .expand(b, self.num_heads, t, n)
                  .reshape(b * self.num_heads, t, n))
        attn_out, _ = self.cross_attn(
            x_norm, kv_norm, kv_norm, key_padding_mask=text_pad_mask, attn_mask=am
        )
        # 参照可能な単語が 1 つも無いフレーム（全 -inf 行）は softmax が NaN になるため 0 に。
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.ln_ff(x)))
        return x


class GPTStereo(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        dff_k: int = 3,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        context_limit: int = -1,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        # Unified stereo layers
        self.layers = nn.ModuleList(
            [
                TransformerStereoLayer(
                    dim=dim,
                    ffn_dim=dff_k * dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    context_limit=context_limit,
                )
                for _ in range(num_layers)
            ]
        )

        # Combinator
        self.combinator = Combinator(dim=dim)

    def forward(self, x1, x2, attention=False):
        self_attention_weights = []
        cross_attention_weights = []

        for layer in self.layers:
            # Process both channels through the stereo layer
            out1 = layer(x1, x2)  # x1 attends to x2
            out2 = layer(x2, x1)  # x2 attends to x1
            x1 = out1["x"]
            x2 = out2["x"]

            if attention:
                self_attention_weights.append([out1["attn"], out2["attn"]])
                cross_attention_weights.append([out1["cross_attn"], out2["cross_attn"]])

        # Combine the channels using the combinator
        combined = self.combinator(x1, x2)

        ret = {"x1": x1, "x2": x2, "x": combined}
        if attention:
            ret["self_attn"] = self_attention_weights
            ret["cross_attn"] = cross_attention_weights
        return ret


def test_gpt():
    """モデル構造を表示"""
    print("=" * 60)
    print("GPT Model Structure")
    print("=" * 60)

    # Test GPT model
    gpt_model = GPT(
        dim=256,
        dff_k=3,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        context_limit=-1,
    )

    print(f"GPT Model Parameters: {sum(p.numel() for p in gpt_model.parameters()):,}")
    print(
        f"Trainable Parameters: {sum(p.numel() for p in gpt_model.parameters() if p.requires_grad):,}"
    )
    print()

    # テスト実行
    print("Testing with sample input...")
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, gpt_model.layers[0].dim)

    with torch.no_grad():
        gpt_result = gpt_model(x, attention=True)

    print(f"GPT Input shape: {x.shape}")
    print(f"GPT Output shape: {gpt_result['x'].shape}")
    print(f"GPT Attention weights shape: {gpt_result['attn'].shape}")
    print()

    # レイヤーごとの詳細
    print("GPT Layer-wise Details:")
    for i, layer in enumerate(gpt_model.layers):
        print(f"Layer {i + 1}:")
        print(f"  Multi-head attention: {layer.dim} -> {layer.dim}")
        print(f"  Heads: {layer.mha.num_heads}")
        print(f"  FFN: {layer.dim} -> {layer.ffn_dim} -> {layer.dim}")

    print("=" * 60)


def test_gpt_stereo():
    """モデル構造を表示"""
    print("=" * 60)
    print("GPT Stereo Model Structure")
    print("=" * 60)

    # Test GPTStereo model
    gpt_stereo_model = GPTStereo(
        dim=256,
        dff_k=3,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        context_limit=-1,
    )

    print(
        f"GPT Stereo Model Parameters: {sum(p.numel() for p in gpt_stereo_model.parameters()):,}"
    )
    print(
        f"Trainable Parameters: {sum(p.numel() for p in gpt_stereo_model.parameters() if p.requires_grad):,}"
    )
    print()

    # テスト実行
    print("Testing with sample input...")
    batch_size = 2
    seq_len = 10
    x1 = torch.randn(batch_size, seq_len, gpt_stereo_model.dim)
    x2 = torch.randn(batch_size, seq_len, gpt_stereo_model.dim)

    with torch.no_grad():
        gpt_stereo_result = gpt_stereo_model(x1, x2, attention=True)

    print(f"GPT Stereo Input shape (Channel 1): {x1.shape}")
    print(f"GPT Stereo Input shape (Channel 2): {x2.shape}")
    print(f"GPT Stereo Output shape (Channel 1): {gpt_stereo_result['x1'].shape}")
    print(f"GPT Stereo Output shape (Channel 2): {gpt_stereo_result['x2'].shape}")
    print(f"GPT Stereo Combined Output shape: {gpt_stereo_result['x'].shape}")
    if "self_attn" in gpt_stereo_result:
        print(
            f"GPT Stereo Self-Attention weights shape (Layer 1, Channel 1): {gpt_stereo_result['self_attn'][0][0].shape}"
        )
        print(
            f"GPT Stereo Self-Attention weights shape (Layer 1, Channel 2): {gpt_stereo_result['self_attn'][0][1].shape}"
        )
    if "cross_attn" in gpt_stereo_result:
        print(
            f"GPT Stereo Cross-Attention weights shape (Layer 1, Channel 1 to Channel 2): {gpt_stereo_result['cross_attn'][0][0].shape}"
        )
        print(
            f"GPT Stereo Cross-Attention weights shape (Layer 1, Channel 2 to Channel 1): {gpt_stereo_result['cross_attn'][0][1].shape}"
        )
    print()


if __name__ == "__main__":
    test_gpt()
    test_gpt_stereo()
