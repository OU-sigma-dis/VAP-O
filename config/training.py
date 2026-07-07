"""学習・データの設定。"""

from dataclasses import dataclass


@dataclass
class OptConfig:
    """最適化の設定。"""

    learning_rate: float = 1e-4
    find_learning_rate: bool = False
    betas = [0.9, 0.999]
    weight_decay: float = 0.01
    accumulate_grad_batches: int = 4  # 実効バッチサイズ = batch_size × accumulate

    # OneCycleLR: warmup → cosine decay
    lr_pct_start: float = 0.1  # 全ステップのうちウォームアップに使う割合
    lr_div_factor: float = 25.0  # 初期LR = max_lr / div_factor
    lr_final_div_factor: float = 1000.0  # 最終LR = max_lr / final_div_factor

    # Early stopping（onset loss のみ監視）
    early_stopping_patience: int = 10
    monitor: str = "val_loss_onset_epoch"
    mode: str = "min"
    save_top_k: int = -1
    max_epochs: int = 100

    saved_dir: str = "./output/checkpoints"

    @staticmethod
    def add_argparse_args(parser):
        for k, v in OptConfig.__dataclass_fields__.items():
            parser.add_argument(f"--opt_{k}", type=v.type, default=v.default)
        return parser

    @staticmethod
    def args_to_conf(args):
        return OptConfig(
            **{
                k.replace("opt_", ""): v
                for k, v in vars(args).items()
                if k.startswith("opt_")
            }
        )


@dataclass
class DataConfig:
    """データセット・データローダーの設定。"""

    train_path: str = "../data/switchboard/vap-o_dataset/train.csv"
    val_path: str = "../data/switchboard/vap-o_dataset/val.csv"
    test_path: str = "../data/switchboard/vap-o_dataset/test.csv"
    batch_size: int = 64
    num_workers: int = 16

    sample_rate: int = 16000
    pin_memory: bool = False
    window_size: float = 20.0
    stride: float = 5.0
    val_stride: float = 20.0

    test_perturbation: int = 0
    # 0: no perturbation
    # 1: flat pitch
    # 2: low pass

    @staticmethod
    def add_argparse_args(parser):
        for k, v in DataConfig.__dataclass_fields__.items():
            parser.add_argument(f"--data_{k}", type=v.type, default=v.default)
        return parser

    @staticmethod
    def args_to_conf(args):
        return DataConfig(
            **{
                k.replace("data_", ""): v
                for k, v in vars(args).items()
                if k.startswith("data_")
            }
        )
