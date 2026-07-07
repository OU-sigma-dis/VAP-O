"""GloVe事前学習済み単語埋め込み + GRUによる軽量テキストエンコーダ。

単語トークン列を受け取り、各位置の隠れ状態ベクトルを返す。
GloVe埋め込みは凍結し、GRUのみを学習する。
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class TextEncoder(nn.Module):
    """GloVe埋め込み + GRUによる軽量テキストエンコーダ。

    入力: 単語トークンID列 (B, seq_len)
    出力: 各位置の隠れ状態ベクトル (B, seq_len, output_dim)
    役割: 事前学習済みGloVe埋め込みで単語の意味を捉え、
          GRUで直近の対話文脈を要約する。
          GloVe埋め込みは凍結、GRUのみ学習対象。

    Attributes:
        embedding: GloVe事前学習済み埋め込み層（凍結）。
        gru: 単方向GRU（文脈モデリング）。
        projection: GRU出力をモデル次元に射影する線形層。
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        glove_dim: int = 100,
        glove_path: str = "",
        vocab_path: str = "",
        dropout: float = 0.1,
        **kwargs,
    ):
        """TextEncoderを初期化する。

        入力:
          vocab_size: 語彙サイズ（PADトークン含む）
          dim: 出力次元数（モデルのdim）
          glove_dim: GloVe埋め込みの次元数（50/100/200/300）
          glove_path: GloVeファイルのパス（例: glove.6B.100d.txt）
          vocab_path: vocab.jsonのパス
          dropout: ドロップアウト率
        出力: なし
        役割: 埋め込み層・GRU・射影層を構築し、GloVeの重みをロードする
        """
        super().__init__()
        self.dim = dim
        self.glove_dim = glove_dim

        # GloVe埋め込み（凍結）
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=glove_dim,
            padding_idx=0,
        )

        # GloVeの重みをロード（ファイルが存在する場合のみ）
        if glove_path and vocab_path:
            from pathlib import Path
            if Path(glove_path).exists() and Path(vocab_path).exists():
                self._load_glove_weights(glove_path, vocab_path)
            else:
                logger.warning(
                    f"GloVeファイルが見つかりません: {glove_path} "
                    "→ ランダム初期化の埋め込みを使用（凍結）"
                )

        # 埋め込みを凍結
        self.embedding.weight.requires_grad = False

        # 1層GRU（単方向）
        self.gru = nn.GRU(
            input_size=glove_dim,
            hidden_size=dim,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )

        self.dropout = nn.Dropout(p=dropout)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"TextEncoder initialized: GloVe {glove_dim}d + GRU → {dim}d, "
            f"params={total:,} (trainable={trainable:,})"
        )

    def _load_glove_weights(self, glove_path: str, vocab_path: str):
        """GloVeファイルからvocab内の単語の埋め込みベクトルをロードする。

        入力:
          glove_path: GloVeテキストファイルのパス
          vocab_path: vocab.jsonのパス
        出力: なし（self.embeddingの重みを更新）
        役割: 語彙内の単語にGloVeベクトルを割り当て、
              未知語はランダム初期化のまま残す
        """
        with open(vocab_path) as f:
            vocab = json.load(f)

        # GloVeベクトルを読み込み
        glove_vectors = {}
        with open(glove_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word in vocab:
                    vec = np.array(parts[1:], dtype=np.float32)
                    if len(vec) == self.glove_dim:
                        glove_vectors[word] = vec

        # 埋め込み行列にGloVeベクトルをセット
        matched = 0
        weight = self.embedding.weight.data
        for word, idx in vocab.items():
            if idx == 0:  # PADはゼロのまま
                continue
            if word in glove_vectors:
                weight[idx] = torch.from_numpy(glove_vectors[word])
                matched += 1

        coverage = matched / (len(vocab) - 1) * 100  # PAD除く
        logger.info(
            f"GloVe loaded: {matched}/{len(vocab)-1} words matched ({coverage:.1f}%)"
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """トークンID列をエンコードし、各位置の隠れ状態を返す。

        入力: token_ids (B, seq_len) のトークンID。0はPADトークン。
        出力: (B, seq_len, dim) の各位置における隠れ状態ベクトル。
        役割: GloVe埋め込み → GRU → 各位置の文脈化された表現を返す
        """
        embedded = self.embedding(token_ids)  # (B, seq_len, glove_dim)
        embedded = self.dropout(embedded)
        output, _ = self.gru(embedded)  # (B, seq_len, dim)
        return output


if __name__ == "__main__":
    from pathlib import Path

    vocab_size = 20521
    batch_size = 4
    seq_len = 32

    # GloVeなしで動作確認
    encoder = TextEncoder(vocab_size=vocab_size, dim=256, glove_dim=100)

    dummy_tokens = torch.randint(low=1, high=vocab_size, size=(batch_size, seq_len))
    output = encoder(dummy_tokens)

    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(f"Input shape:      {dummy_tokens.shape}")
    print(f"Output shape:     {output.shape}")
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
