"""モデルの設定。"""

from dataclasses import dataclass


def field_names(dataclass_cls):
    return [f.name for f in dataclass_cls.__dataclass_fields__.values()]


@dataclass
class VapConfig:
    """VapGPTモデルの設定を管理するデータクラス。"""

    # General
    sample_rate: int = 16_000
    frame_hz: int = 20

    # Audio Encoder (CPC)
    cpc_model_pt: str = "default"
    freeze_encoder: bool = True
    load_pretrained: bool = True

    # Model dimensions
    dim: int = 256

    # Text Encoder (GloVe + Cross-Attention)
    use_text: bool = True  # テキスト入力を使用するか
    # True: 各音声フレーム t は position <= t の単語のみ参照（causal, オンライン整合）
    # False: 窓内の全単語を参照（非因果, 未来トークンをリークする従来設定）
    text_causal: bool = True
    num_text_cross_layers: int = 2  # テキスト Cross-Attention 層数
    vocab_size: int = 20521
    glove_dim: int = 100
    glove_path: str = "../data/glove/glove.6B.100d.txt"
    vocab_path: str = "../data/switchboard/vap-o_dataset/vocab.json"
    max_text_tokens: int = 64
    text_dropout: float = 0.1

    # Onset Proximity
    onset_horizon: float = 3.0  # onset proximity の正規化基準（秒）

    # Filler Classification
    num_filler_classes: int = 8
    use_filler: bool = True  # False: フィラー予測を損失から外す（補助ヘッド無効化）

    # Transformer
    use_stereo: bool = False  # True: チャネル分離(GPT+GPTStereo), False: 結合(GPT単体)
    num_channel_layers: int = 1  # チャネル別 Self-Attention 層数（stereo 時のみ）
    num_cross_layers: int = 3  # チャネル間 Cross-Attention 層数（stereo 時のみ）
    num_transformer_layers: int = 4  # 結合モード時の GPT 層数
    num_heads: int = 4
    transformer_dff_k: int = 3  # FFN dim = dim * dff_k
    transformer_dropout: float = 0.25

    # Fusion
    fusion_dropout: float = 0.1

    # CPC固有設定
    context_limit_cpc_sec: float = -1

    @staticmethod
    def add_argparse_args(parser):
        """argparse.ArgumentParserにこのクラスのフィールドを追加する。"""
        for f in field_names(VapConfig):
            default = VapConfig.__dataclass_fields__[f].default
            if isinstance(default, bool):
                parser.add_argument(
                    f"--vap_{f}", type=lambda x: bool(int(x)), default=default
                )
            else:
                field_type = VapConfig.__dataclass_fields__[f].type
                parser.add_argument(f"--vap_{f}", type=field_type, default=default)
        return parser, []

    @staticmethod
    def from_args(args):
        """argparseでパースされたargsからVapConfigインスタンスを作成する。"""
        arg_dict = {
            k.replace("vap_", ""): v
            for k, v in vars(args).items()
            if k.startswith("vap_")
        }
        return VapConfig(**arg_dict)

    @staticmethod
    def args_to_conf(args):
        """from_argsのエイリアス。"""
        return VapConfig.from_args(args)
