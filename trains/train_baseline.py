"""ベースライン VAP モデル（Inoue アーキテクチャ）を Switchboard で1から学習するスクリプト。

MaAI の VapGPT（256クラス分類）を使用し、提案手法と同じデータ分割・学習条件で訓練する。
これにより、アーキテクチャの差異のみを公平に比較できる。

使い方:
  python trains/train_baseline.py
"""

import json
import os
import sys
import warnings
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger

# MaAI の参照実装を使用する。公開リポジトリにはコピーしないため、利用者が
# VAPO_MAAI_ROOT に MaAI checkout のルートを指定する。
_ROOT = Path(__file__).resolve().parent.parent
_maai_root = Path(os.environ.get("VAPO_MAAI_ROOT", "")).expanduser()
if not (_maai_root / "train" / "model.py").is_file() or not (_maai_root / "src" / "maai").is_dir():
    raise RuntimeError(
        "Set VAPO_MAAI_ROOT to a MaAI checkout containing train/model.py and src/maai/."
    )
sys.path.insert(0, str(_maai_root / "train"))
sys.path.insert(0, str(_maai_root / "src"))

# The reference package imports optional audio dependencies that are not needed here.
import types
maai_fake = types.ModuleType("maai")
maai_fake.__path__ = [str(_maai_root / "src" / "maai")]
maai_fake.__package__ = "maai"
sys.modules["maai"] = maai_fake

from model import VapGPT as BaselineVapGPT, VapConfig as BaselineVapConfig
from objective import ObjectiveVAP

# VAP-O のデータパイプライン
sys.path.insert(0, str(_ROOT))
from datasets.datamodule import VapDataModule

warnings.filterwarnings("ignore", message=".*litmodels.*")
warnings.filterwarnings("ignore", message=".*tensorboardX.*")

torch.set_float32_matmul_precision("medium")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.set_default_dtype(torch.float32)


class BaselineVAPModel(BaselineVapGPT, pl.LightningModule):
    """MaAI VapGPT の Lightning ラッパー。提案手法と同じ条件で学習。"""

    def __init__(self, conf=None, learning_rate=1e-4, weight_decay=0.01):
        if conf is None:
            conf = BaselineVapConfig(
                frame_hz=20,
                bin_times=[0.2, 0.4, 0.6, 0.8],
                cpc_model_pt=os.environ.get(
                    "VAPO_CPC_CHECKPOINT", "assets/checkpoints/cpc/60k_epoch4-d0f474de.pt"
                ),
                freeze_encoder=1,
                load_pretrained=1,
                channel_layers=1,
                cross_layers=3,
                num_heads=4,
                dim=256,
                dropout=0.1,
            )
        super().__init__(conf)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward_from_cpc_features(self, cpc_feat_1, cpc_feat_2):
        """事前抽出済み CPC 特徴量（100Hz）からの forward パス。

        提案手法と同じ CPC 特徴量を流用し、ダウンサンプル（100Hz→20Hz）してから
        Transformer に通す。
        """
        # ダウンサンプル（100Hz → 20Hz）
        x1 = self.encoder.downsample(cpc_feat_1)
        x2 = self.encoder.downsample(cpc_feat_2)

        # Transformer
        o1 = self.ar_channel(x1)
        o2 = self.ar_channel(x2)
        out = self.ar(o1["x"], o2["x"])

        # 出力ヘッド
        v1 = self.va_classifier(out["x1"])
        v2 = self.va_classifier(out["x2"])
        vad = torch.cat((v1, v2), dim=-1)
        logits = self.vap_head(out["x"])

        return {"logits": logits, "vad": vad}

    def shared_step(self, batch, reduction="mean"):
        """提案手法の DataLoader 出力に対応。"""
        va = batch["va"]  # (B, T, 2)
        labels = self.objective.get_labels(va)

        # 事前抽出済み CPC 特徴量があればそれを使用（高速）
        cpc_feat_1 = batch.get("cpc_feat_1")
        cpc_feat_2 = batch.get("cpc_feat_2")
        if cpc_feat_1 is not None and cpc_feat_2 is not None:
            out = self.forward_from_cpc_features(cpc_feat_1, cpc_feat_2)
        else:
            out = self(waveform=batch["waveform"])

        # フレーム数を合わせる
        t_model = out["logits"].shape[1]
        t_label = labels.shape[1]
        t_min = min(t_model, t_label)

        out["vap_loss"] = self.objective.loss_vap(
            out["logits"][:, :t_min], labels[:, :t_min], reduction=reduction
        )
        out["vad_loss"] = self.vad_loss(out["vad"][:, :t_min], va[:, :t_min])

        return out

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=[0.9, 0.999],
            weight_decay=self.weight_decay,
        )
        # OneCycleLR（提案手法と同じ）
        total_steps = self.trainer.estimated_stepping_batches
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                pct_start=0.1,
                div_factor=25.0,
                final_div_factor=1000.0,
                anneal_strategy="cos",
            ),
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": opt, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx, **kwargs):
        out = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]

        loss = out["vap_loss"] + out["vad_loss"]

        self.log("loss_train_vap", out["vap_loss"], batch_size=batch_size, on_epoch=True, on_step=True, sync_dist=True)
        self.log("loss_train_vad", out["vad_loss"], batch_size=batch_size, on_epoch=True, on_step=True, sync_dist=True)
        self.log("batch_loss", loss, batch_size=batch_size, on_step=True, sync_dist=True)

        # 学習率ログ
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", current_lr, on_step=True, sync_dist=True)

        if batch_idx % 100 == 0:
            logger.info(
                f"Epoch {self.current_epoch}, Batch {batch_idx}: "
                f"Loss={loss:.4f} (VAP: {out['vap_loss']:.4f}, VAD: {out['vad_loss']:.4f}), "
                f"LR={current_lr:.6f}"
            )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx, **kwargs):
        out = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]

        self.log("val_loss_vap", out["vap_loss"], batch_size=batch_size, on_epoch=True, sync_dist=True)
        self.log("val_loss_vad", out["vad_loss"], batch_size=batch_size, on_epoch=True, sync_dist=True)
        val_total = out["vap_loss"] + out["vad_loss"]
        self.log("val_loss", val_total, batch_size=batch_size, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self, *_):
        val_vap = self.trainer.callback_metrics.get("val_loss_vap", 0.0)
        val_vad = self.trainer.callback_metrics.get("val_loss_vad", 0.0)
        val_total = self.trainer.callback_metrics.get("val_loss", 0.0)
        logger.info(
            f"Validation Epoch {self.current_epoch}: "
            f"Total={val_total:.4f} (VAP={val_vap:.4f}, VAD={val_vad:.4f})"
        )


def train(seed: int = 42, save_dir: str | None = None,
          train_csv: str | None = None, val_csv: str | None = None):
    # 複数 seed 検証（査読対応）: seed は学習の乱数（重み初期化・シャッフル等）のみを
    # 変える。データ分割は CSV で固定済みであり seed に依存しない。
    pl.seed_everything(seed)

    # ============================
    # 学習条件（提案手法と統一）
    # ============================
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    BATCH_SIZE = 64
    ACCUMULATE_GRAD_BATCHES = 4
    MAX_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 10
    WINDOW_SIZE = 20.0
    STRIDE = 5.0
    VAL_STRIDE = 20.0
    NUM_WORKERS = 16

    # パス
    TRAIN_PATH = train_csv or "../data/switchboard/vap-o_dataset/train.csv"
    VAL_PATH = val_csv or "../data/switchboard/vap-o_dataset/val.csv"
    SAVE_DIR = save_dir or "./output/checkpoints_baseline_retrain"
    LOG_DIR = "./output/logs"

    os.chdir(Path(__file__).resolve().parent.parent)

    # モデル
    model = BaselineVAPModel(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Baseline VapGPT: {total_params:,} params ({trainable_params:,} trainable)")

    # データ
    # 事前抽出済み CPC 特徴量を使用（提案手法と同じ特徴量を流用）
    CPC_FEATURE_DIR = str(Path(TRAIN_PATH).parent / "cpc_features")
    dm = VapDataModule(
        train_path=TRAIN_PATH,
        val_path=VAL_PATH,
        test_path=None,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        val_stride=VAL_STRIDE,
        frame_hz=20,
        max_text_tokens=64,
        cpc_feature_dir=CPC_FEATURE_DIR,
    )
    dm.prepare_data()

    # ログ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"baseline_retrain_seed{seed}_{timestamp}"
    log_dir = Path(LOG_DIR) / name
    log_dir.mkdir(parents=True, exist_ok=True)

    # 実験設定を保存
    config = {
        "name": name,
        "seed": seed,
        "model": "MaAI VapGPT (256-class)",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "accumulate_grad_batches": ACCUMULATE_GRAD_BATCHES,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "val_stride": VAL_STRIDE,
        "architecture": {
            "frame_hz": 20,
            "dim": 256,
            "channel_layers": 1,
            "cross_layers": 3,
            "num_heads": 4,
            "dropout": 0.1,
            "bin_times": [0.2, 0.4, 0.6, 0.8],
            "n_classes": 256,
            "encoder": "CPC (frozen)",
        },
    }
    with open(log_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.add(log_dir / "train.log", rotation="100 MB")
    logger.info(f"Config saved: {log_dir / 'config.json'}")

    csv_logger = CSVLogger(save_dir=str(log_dir), name="metrics")

    # コールバック
    os.makedirs(SAVE_DIR, exist_ok=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=SAVE_DIR,
            mode="min",
            monitor="val_loss",
            save_top_k=-1,
            auto_insert_metric_name=False,
            filename="baseline_retrain-epoch{epoch}-val_{val_loss:.5f}",
        ),
        RichProgressBar(),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            strict=True,
            verbose=True,
        ),
    ]

    # デバイス
    if torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    elif torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1

    # Trainer
    trainer = pl.Trainer(
        logger=csv_logger,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        precision=32,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        max_epochs=MAX_EPOCHS,
    )

    logger.info(f"Accelerator: {accelerator}, Accumulate: {ACCUMULATE_GRAD_BATCHES}")
    logger.info("Training started...")
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default=None)
    ap.add_argument("--train_csv", type=str, default=None)
    ap.add_argument("--val_csv", type=str, default=None)
    args = ap.parse_args()
    train(seed=args.seed, save_dir=args.save_dir,
          train_csv=args.train_csv, val_csv=args.val_csv)
