import json
import os
from argparse import ArgumentParser
from datetime import datetime
from os import environ, path
from pathlib import Path
from typing import Dict

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from torchmetrics.classification import Accuracy, F1Score

from datasets.datamodule import VapDataModule
from config import VapConfig, OptConfig, DataConfig, EventConfig
from models.model import VapGPT
from vap.events import TurnTakingEvents

torch.set_float32_matmul_precision("medium")

# Lightning の不要な広告メッセージを抑制
import warnings
warnings.filterwarnings("ignore", message=".*litmodels.*")
warnings.filterwarnings("ignore", message=".*tensorboardX.*")

# MPSデバイス使用時はfloat32をデフォルトにする
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.set_default_dtype(torch.float32)


def get_device_auto():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_args():
    parser = ArgumentParser("VoiceActivityProjection")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    # parser = pl.Trainer.add_argparse_args(parser)
    parser = OptConfig.add_argparse_args(parser)
    parser = DataConfig.add_argparse_args(parser)
    parser, fields_added = VapConfig.add_argparse_args(parser)
    parser, fields_added = EventConfig.add_argparse_args(parser, fields_added)
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Device to use for training (auto/cuda/cpu/mps)",
    )
    parser.add_argument("--devices", type=str, default="0")
    args = parser.parse_args()

    model_conf = VapConfig.args_to_conf(args)
    opt_conf = OptConfig.args_to_conf(args)
    data_conf = DataConfig.args_to_conf(args)
    event_conf = EventConfig.args_to_conf(args)

    # Remove all non trainer args
    cfg_dict = vars(args)
    for k, _ in list(cfg_dict.items()):
        if (
            k.startswith("data_")
            or k.startswith("vap_")
            or k.startswith("opt_")
            or k.startswith("event_")
        ):
            cfg_dict.pop(k)

    return {
        "args": args,
        "cfg_dict": cfg_dict,
        "model": model_conf,
        "event": event_conf,
        "opt": opt_conf,
        "data": data_conf,
    }


def get_run_name(configs) -> str:
    m = configs["model"]
    s = "VapGPT"
    s += f"_{m.frame_hz}Hz"
    s += "_cpc"
    s += f"_glove{m.glove_dim}gru"
    s += f"_d{m.dim}"
    return s


def train() -> None:
    configs = get_args()
    cfg_dict = configs["cfg_dict"]

    pl.seed_everything(cfg_dict["seed"])
    local_rank = environ.get("LOCAL_RANK", 0)

    model = VAPModel(
        configs["model"], opt_conf=configs["opt"], event_conf=configs["event"]
    )

    name = get_run_name(configs)

    dconf = configs["data"]

    # MPSデバイスでは pin_memory は無効にする
    pin_memory = dconf.pin_memory
    device = (
        configs["args"].device
        if configs["args"].device != "auto"
        else get_device_auto()
    )
    if device == "mps":
        pin_memory = False

    dm = VapDataModule(
        train_path=dconf.train_path,
        val_path=dconf.val_path,
        test_path=None,
        batch_size=dconf.batch_size,
        num_workers=dconf.num_workers,
        pin_memory=pin_memory,
        window_size=dconf.window_size,
        stride=dconf.stride,
        val_stride=dconf.val_stride,
        frame_hz=configs["model"].frame_hz,
        max_text_tokens=configs["model"].max_text_tokens,
        cpc_feature_dir=str(Path(dconf.train_path).parent / "cpc_features"),
    )
    dm.prepare_data()

    if configs["args"].devices is None:
        gpu_devices = -1
    else:
        gpu_devices = [int(d.strip()) for d in configs["args"].devices.split(",")]

    oconf = configs["opt"]

    # ログディレクトリの設定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(oconf.saved_dir).parent / "logs" / f"{name}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 実験設定を保存
    experiment_config = {
        "name": name,
        "timestamp": timestamp,
        "model": {k: v for k, v in configs["model"].__dict__.items() if not k.startswith("_")},
        "optimizer": {k: v for k, v in oconf.__dict__.items() if not k.startswith("_")},
        "data": {k: v for k, v in dconf.__dict__.items() if not k.startswith("_")},
        "event": {k: v for k, v in configs["event"].__dict__.items() if not k.startswith("_")},
        "seed": cfg_dict["seed"],
        "device": device,
    }
    with open(log_dir / "config.json", "w") as f:
        json.dump(experiment_config, f, indent=2, default=str)
    logger.info(f"実験設定を保存: {log_dir / 'config.json'}")

    # loguruのファイルログも追加
    logger.add(log_dir / "train.log", rotation="100 MB")

    # CSV Logger（PyTorch Lightning）
    csv_logger = CSVLogger(
        save_dir=str(log_dir),
        name="metrics",
    )

    # チェックポイント保存ディレクトリ
    os.makedirs(oconf.saved_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=oconf.saved_dir,
            mode=oconf.mode,
            monitor=oconf.monitor,
            save_top_k=oconf.save_top_k,
            auto_insert_metric_name=False,
            filename=name + "-epoch{epoch}-onset_{val_loss_onset_epoch:.5f}",
        ),
        RichProgressBar(),
    ]

    callbacks.append(
        EarlyStopping(
            monitor=oconf.monitor,
            mode=oconf.mode,
            patience=oconf.early_stopping_patience,
            strict=True,
            verbose=True,
        )
    )

    if local_rank == 0:
        logger.info(f"Early stopping: monitor={oconf.monitor}, patience={oconf.early_stopping_patience}")

    # デバイス設定
    device = (
        configs["args"].device
        if configs["args"].device != "auto"
        else get_device_auto()
    )

    if device == "cuda":
        accelerator = "gpu"
        devices = gpu_devices
        precision = "16-mixed"
    elif device == "mps":
        accelerator = "mps"
        devices = 1
        # MPS では GradScaler 非対応のため 32bit を使用
        precision = 32
    else:
        accelerator = "cpu"
        devices = 1
        precision = 32

    # LR Finder
    if oconf.find_learning_rate:
        logger.info("=== LR Finder 実行中 ===")
        from pytorch_lightning.tuner import Tuner
        lr_trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
        )
        tuner = Tuner(lr_trainer)
        model.learning_rate = oconf.learning_rate
        lr_finder = tuner.lr_find(model, dm, attr_name="learning_rate")
        suggested_lr = lr_finder.suggestion()
        if suggested_lr is not None:
            logger.info(f"LR Finder 推奨値: {suggested_lr:.6f} (元: {oconf.learning_rate:.6f})")
            oconf.learning_rate = suggested_lr
            # config.json に記録
            experiment_config["optimizer"]["learning_rate_found"] = suggested_lr
            with open(log_dir / "config.json", "w") as f:
                json.dump(experiment_config, f, indent=2, default=str)
        else:
            logger.warning("LR Finder が推奨値を見つけられませんでした。デフォルトLRを使用します。")
        # LR Finder で作られた内部状態をリセット
        model = VAPModel(
            configs["model"], opt_conf=oconf, event_conf=configs["event"]
        )
        model.learning_rate = oconf.learning_rate
    else:
        logger.info(f"Learning Rate: {oconf.learning_rate}")

    # Trainer構築
    trainer_kwargs = {
        "logger": csv_logger,
        "callbacks": callbacks,
        "accelerator": accelerator,
        "devices": devices,
        "precision": precision,
        "accumulate_grad_batches": oconf.accumulate_grad_batches,
        "max_epochs": oconf.max_epochs,
    }

    if accelerator not in ("mps", "cpu"):
        bool_find_unused_parameters = configs["model"].context_limit_cpc_sec > 0
        trainer_kwargs["strategy"] = DDPStrategy(
            find_unused_parameters=bool_find_unused_parameters
        )

    trainer = pl.Trainer(**trainer_kwargs)
    logger.info(f"Precision: {precision}, Accumulate grad batches: {oconf.accumulate_grad_batches}")

    trainer.fit(model, datamodule=dm)


# Used in training (LightningModule) but not required for inference
class VAPModel(VapGPT, pl.LightningModule):
    def __init__(self, conf, opt_conf=None, event_conf=None):
        super().__init__(conf)

        self.opt_conf = opt_conf
        self.event_conf = event_conf

        # Training params
        self.save_hyperparameters()

        # Metrics
        self.event_extractor = None
        if event_conf is not None:
            self.event_extractor = TurnTakingEvents(event_conf)

        self.training_step_outputs = []
        # Hold/Shift metrics
        self.test_hs_reference = [0, 0]
        self.test_hs_confusion_matrix = {
            "ref=0:pre=0": 0,
            "ref=0:pre=1": 0,
            "ref=1:pre=0": 0,
            "ref=1:pre=1": 0,
        }

        # Shift prediction metrics
        self.test_pred_shift_reference = [0, 0]
        self.test_pred_shift_confusion_matrix = {
            "ref=0:pre=0": 0,
            "ref=0:pre=1": 0,
            "ref=1:pre=0": 0,
            "ref=1:pre=1": 0,
        }

        # Backchannel metrics
        self.test_bc_reference = [0, 0]
        self.test_bc_confusion_matrix = {
            "ref=0:pre=0": 0,
            "ref=0:pre=1": 0,
            "ref=1:pre=0": 0,
            "ref=1:pre=1": 0,
        }

    def get_metrics(self):
        metrics = {"acc": {}, "f1": {}}

        ACC_TASK = "multiclass"
        ACC_AVERAGE = "none"

        metrics["acc"]["hs"] = Accuracy(
            task=ACC_TASK, num_classes=2, average=ACC_AVERAGE
        ).to(self.device)

        metrics["acc"]["sp"] = Accuracy(
            task=ACC_TASK, num_classes=2, average=ACC_AVERAGE
        ).to(self.device)

        metrics["f1"]["hs"] = F1Score(
            task="multiclass",
            num_classes=2,
            average="weighted",
        ).to(self.device)

        metrics["f1"]["sp"] = F1Score(
            task="multiclass",
            num_classes=2,
            average="weighted",
        ).to(self.device)

        metrics["acc"]["sp_ov"] = Accuracy(
            task=ACC_TASK, num_classes=2, average=ACC_AVERAGE
        ).to(self.device)

        metrics["f1"]["sp_ov"] = F1Score(
            task="multiclass",
            num_classes=2,
            average="weighted",
        ).to(self.device)

        metrics["acc"]["bc"] = Accuracy(
            task=ACC_TASK, num_classes=2, average=ACC_AVERAGE
        ).to(self.device)

        metrics["f1"]["bc"] = F1Score(
            task="multiclass",
            num_classes=2,
            average="weighted",
        ).to(self.device)

        # フィラー分類 Top-K 正解率（class 0=pad を除外するため ignore_index 指定）
        num_filler = self.conf.num_filler_classes
        metrics["acc"]["filler_top1"] = Accuracy(
            task="multiclass",
            num_classes=num_filler,
            average="macro",
            top_k=1,
            ignore_index=0,
        ).to(self.device)

        metrics["acc"]["filler_top3"] = Accuracy(
            task="multiclass",
            num_classes=num_filler,
            average="macro",
            top_k=3,
            ignore_index=0,
        ).to(self.device)

        return metrics

    def _get_filler_class_weights(self, device):
        """フィラークラスの逆頻度重みを返す。クラス不均衡の補正に使用。"""
        if not hasattr(self, "_filler_weights"):
            # filler_categories.json から読み込み、またはハードコード
            # 平方根で緩和した逆頻度重み（オーバーフィット防止）
            weights = [0.0, 0.39, 1.53, 1.49, 5.72, 2.68, 1.80, 8.66]
            self._filler_weights = torch.tensor(weights, dtype=torch.float32)
        return self._filler_weights.to(device)

    @torch.no_grad()
    def extract_predictions(
        self,
        onset_proximity: torch.Tensor,
        events: Dict,
    ):
        """Onset Proximity出力からイベント評価用の予測・ターゲットを抽出する。

        各イベント領域内のonset proximityのmax値を予測として使う（閾値交差方式）。

        Args:
            onset_proximity: (B, T, 2) 各話者のonset proximity
            events: TurnTakingEventsが返すイベント辞書

        Returns:
            (preds, targets): 各メトリクスの予測値とターゲットの辞書ペア
        """
        metric_definitions = {
            # neg_flip: negイベントのspeakerが「継続話者」の場合True（1-speakerで相手の値を使う）
            #           negイベントのspeakerが既に「相手/BC話者」の場合False（そのまま使う）
            "hs": {"pos": "shift", "neg": "hold", "neg_flip": True},
            "pred_shift": {"pos": "pred_shift", "neg": "pred_shift_neg", "neg_flip": True},
            "pred_shift_ov": {"pos": "pred_shift_ov", "neg": "pred_shift_ov_neg", "neg_flip": True},
            "bc": {"pos": "pred_backchannel", "neg": "pred_backchannel_neg", "neg_flip": False},
        }

        all_preds, all_targets = {}, {}
        batch_size = onset_proximity.shape[0]

        for metric, conf in metric_definitions.items():
            preds_list, targets_list = [], []
            neg_flip = conf["neg_flip"]

            for key, target_val in [(conf["pos"], 1.0), (conf["neg"], 0.0)]:
                if key not in events:
                    continue
                for b in range(batch_size):
                    for start, end, speaker in events[key][b]:
                        if start >= end or start >= onset_proximity.shape[1]:
                            continue
                        end = min(end, onset_proximity.shape[1])
                        if target_val == 0.0 and neg_flip:
                            sp_idx = 1 - speaker
                        else:
                            sp_idx = speaker
                        # 領域スコア集約。onset proximity は onset に向かって上昇する
                        # ため、hs/pred_shift は「領域の最終フレーム値(=沈黙直前で最も
                        # 情報が多い)」が最良（早期スパイクに強い）。overlap/bc は max。
                        region = onset_proximity[b, start:end, sp_idx]
                        if metric in ("hs", "pred_shift"):
                            pred = region[-1]
                        else:
                            pred = region.max()
                        preds_list.append(pred.unsqueeze(0))
                        targets_list.append(
                            torch.full((1,), target_val, device=onset_proximity.device)
                        )

            all_preds[metric] = (
                torch.cat(preds_list) if preds_list else None
            )
            all_targets[metric] = (
                torch.cat(targets_list).long() if targets_list else None
            )

        return all_preds, all_targets

    def _update_filler_topk_metrics(
        self, out: Dict, batch: Dict, split: str = "val"
    ):
        """フィラー分類のTop-K正解率メトリクスを更新する。

        入力:
          out: モデル出力（filler_logits を含む）
          batch: バッチデータ（next_filler_class を含む）
          split: "val" or "test"
        """
        m = self.val_metrics if split == "val" else self.test_metrics

        t_model = out["filler_logits"].shape[1]
        t_label = batch["next_filler_class"].shape[1]
        t_min = min(t_model, t_label)

        logits = out["filler_logits"][:, :t_min]  # (B, T, num_classes)
        targets = batch["next_filler_class"][:, :t_min]  # (B, T)

        # pad(=0)以外のフレームだけで評価する
        mask = targets != 0
        if mask.sum() == 0:
            return

        logits_flat = logits[mask]  # (N, num_classes)
        targets_flat = targets[mask]  # (N,)

        m["acc"]["filler_top1"].update(preds=logits_flat, target=targets_flat)
        m["acc"]["filler_top3"].update(preds=logits_flat, target=targets_flat)

    def metrics_step(self, preds, targets, split="val"):
        m = self.val_metrics if split == "val" else self.test_metrics

        # The metrics don't work if the predictions are not rounded
        # I don't know why...
        if preds["hs"] is not None:
            if len(targets["hs"]) >= 1:
                m["f1"]["hs"].update(preds=preds["hs"].round(), target=targets["hs"])
                m["acc"]["hs"].update(preds=preds["hs"].round(), target=targets["hs"])

        if preds["pred_shift"] is not None:
            if len(targets["pred_shift"]) >= 1:
                m["f1"]["sp"].update(
                    preds=preds["pred_shift"].round(), target=targets["pred_shift"]
                )
                m["acc"]["sp"].update(
                    preds=preds["pred_shift"].round(), target=targets["pred_shift"]
                )

        if preds.get("pred_shift_ov") is not None:
            if len(targets["pred_shift_ov"]) >= 1:
                m["f1"]["sp_ov"].update(
                    preds=preds["pred_shift_ov"].round(), target=targets["pred_shift_ov"]
                )
                m["acc"]["sp_ov"].update(
                    preds=preds["pred_shift_ov"].round(), target=targets["pred_shift_ov"]
                )

        if preds.get("bc") is not None:
            if len(targets["bc"]) >= 1:
                m["f1"]["bc"].update(preds=preds["bc"].round(), target=targets["bc"])
                m["acc"]["bc"].update(preds=preds["bc"].round(), target=targets["bc"])

    def metrics_epoch(self, split="val"):
        if split == "test":
            # Calculate metrics for all confusion matrices
            def calculate_metrics(confusion_matrix):
                tp = confusion_matrix["ref=1:pre=1"]
                fp = confusion_matrix["ref=0:pre=1"]
                fn = confusion_matrix["ref=1:pre=0"]
                tn = confusion_matrix["ref=0:pre=0"]

                # Balanced accuracy = (TPR + TNR) / 2
                if tp + fn == 0 or tn + fp == 0:
                    balanced_accuracy = 0
                else:
                    balanced_accuracy = (tp / (tp + fn) + tn / (tn + fp)) / 2

                # Precision = TP / (TP + FP)
                precision = tp / (tp + fp) if tp + fp > 0 else 0

                # Recall = TP / (TP + FN)
                recall = tp / (tp + fn) if tp + fn > 0 else 0

                # F1 = 2 * (precision * recall) / (precision + recall)
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision + recall > 0
                    else 0
                )

                return balanced_accuracy, precision, recall, f1

            # Hold/Shift
            hs_balanced_accuracy, hs_precision, hs_recall, hs_f1 = calculate_metrics(
                self.test_hs_confusion_matrix
            )
            self.log(
                f"{split}_hs_balanced_accuracy", hs_balanced_accuracy, sync_dist=True
            )
            self.log(f"{split}_hs_precision", hs_precision, sync_dist=True)
            self.log(f"{split}_hs_recall", hs_recall, sync_dist=True)
            self.log(f"{split}_hs_f1", hs_f1, sync_dist=True)

            # Shift prediction
            (
                pred_shift_balanced_accuracy,
                pred_shift_precision,
                pred_shift_recall,
                pred_shift_f1,
            ) = calculate_metrics(self.test_pred_shift_confusion_matrix)
            self.log(
                f"{split}_pred_shift_balanced_accuracy",
                pred_shift_balanced_accuracy,
                sync_dist=True,
            )
            self.log(
                f"{split}_pred_shift_precision", pred_shift_precision, sync_dist=True
            )
            self.log(f"{split}_pred_shift_recall", pred_shift_recall, sync_dist=True)
            self.log(f"{split}_pred_shift_f1", pred_shift_f1, sync_dist=True)

            # Backchannel
            bc_balanced_accuracy, bc_precision, bc_recall, bc_f1 = calculate_metrics(
                self.test_bc_confusion_matrix
            )
            self.log(f"{split}_bc_balanced_accuracy", bc_balanced_accuracy, sync_dist=True)
            self.log(f"{split}_bc_precision", bc_precision, sync_dist=True)
            self.log(f"{split}_bc_recall", bc_recall, sync_dist=True)
            self.log(f"{split}_bc_f1", bc_f1, sync_dist=True)

        # torchmetricsベースの全メトリクスをcompute → log → reset
        m = self.val_metrics if split == "val" else self.test_metrics
        for metric_type in ["acc", "f1"]:
            if metric_type not in m:
                continue
            for name, metric in m[metric_type].items():
                try:
                    value = metric.compute()
                    if isinstance(value, torch.Tensor) and value.numel() > 1:
                        logger.info(f"  {split}_{metric_type}_{name}: {value.tolist()}")
                    else:
                        self.log(f"{split}_{metric_type}_{name}", float(value), sync_dist=True)
                        logger.info(f"  {split}_{metric_type}_{name}: {float(value):.4f}")
                    metric.reset()
                except Exception:
                    pass

    def shared_step(
        self, batch: Dict, reduction: str = "mean"
    ) -> Dict[str, torch.Tensor]:
        """モデルの損失を計算する。

        Args:
            batch: dict, 'waveform', 'va', 'onset_proximity',
                   'text_tokens', 'text_token_positions', 'next_filler_class' を含む

        Returns:
            out: dict, model出力 + 'onset_loss', 'vad_loss', 'filler_loss'
        """
        n_frames = batch["onset_proximity"].shape[1]
        out = self(
            audio=batch["waveform"],
            text_tokens=batch["text_tokens"],
            text_token_positions=batch["text_token_positions"],
            n_frames=n_frames,
            cpc_feat_1=batch.get("cpc_feat_1"),
            cpc_feat_2=batch.get("cpc_feat_2"),
        )

        # モデル出力とターゲットのフレーム数を揃える
        t_model = out["onset_proximity"].shape[1]
        t_min = min(t_model, n_frames)

        # Onset Proximity 損失: 重み付きMSE
        # 3段階の重み:
        #   沈黙区間（両話者とも非発話）: 10.0 ← shift/hold判断の核心
        #   onset近傍（target>0）:         3.0 ← 次の話者の予測
        #   それ以外:                      1.0
        pred_onset = out["onset_proximity"][:, :t_min]
        target_onset = batch["onset_proximity"][:, :t_min]
        va = batch["va"][:, :t_min]  # (B, T, 2)

        # 沈黙フレーム: 両話者とも非発話
        silence = (va.sum(dim=-1, keepdim=True) == 0).float()  # (B, T, 1)
        # 話者間の差が大きいフレーム: 一方がonset近傍で他方がゼロ
        speaker_diff = (target_onset[:, :, 0:1] - target_onset[:, :, 1:2]).abs()  # (B, T, 1)

        onset_weight = torch.ones_like(target_onset)
        onset_weight = torch.where(target_onset > 0, 2.0, onset_weight)
        onset_weight = onset_weight + 3.0 * silence.expand_as(onset_weight)
        onset_weight = onset_weight + 2.0 * speaker_diff.expand_as(onset_weight)
        onset_loss = (onset_weight * (pred_onset - target_onset) ** 2).mean()

        # VAD損失: BCE(pred_vad, target_va)
        vad_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out["vad"][:, :t_min], batch["va"][:, :t_min]
        )

        # フィラー分類損失: クラス重み付きCrossEntropy
        filler_loss = torch.nn.functional.cross_entropy(
            out["filler_logits"][:, :t_min].reshape(-1, self.conf.num_filler_classes),
            batch["next_filler_class"][:, :t_min].reshape(-1),
            ignore_index=0,  # PAD(=0)を無視
            weight=self._get_filler_class_weights(out["filler_logits"].device),
        )

        # 話者間コントラスト損失:
        # GT で話者間の差が大きいフレームで、pred にも差を要求する
        gt_diff = (target_onset[:, :, 0] - target_onset[:, :, 1]).abs()  # (B, T)
        pred_diff = (pred_onset[:, :, 0] - pred_onset[:, :, 1]).abs()    # (B, T)
        # GT で差が 0.3 以上あるフレームを対象
        contrast_mask = gt_diff > 0.3
        if contrast_mask.sum() > 0:
            margin = 0.3
            contrast_loss = torch.clamp(margin - pred_diff[contrast_mask], min=0).mean()
        else:
            contrast_loss = torch.tensor(0.0, device=pred_onset.device)

        # 差分パターン一致損失:
        # GT の時間変化パターンに pred を合わせる（ノイズ抑制 + 急落の再現）
        gt_temporal_diff = target_onset[:, 1:] - target_onset[:, :-1]   # (B, T-1, 2)
        pred_temporal_diff = pred_onset[:, 1:] - pred_onset[:, :-1]     # (B, T-1, 2)
        diff_loss = ((pred_temporal_diff - gt_temporal_diff) ** 2).mean()

        out["onset_loss"] = onset_loss
        out["contrast_loss"] = contrast_loss
        out["diff_loss"] = diff_loss
        out["vad_loss"] = vad_loss
        out["filler_loss"] = filler_loss

        return out

    def configure_optimizers(self) -> Dict:
        assert self.opt_conf is not None, "configure_optimizers: No Opt conf!"
        # LR Finder が learning_rate を更新している場合はそちらを使用
        lr = getattr(self, "learning_rate", self.opt_conf.learning_rate)
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=self.opt_conf.betas,
            weight_decay=self.opt_conf.weight_decay,
        )

        # OneCycleLR: warmup（上昇）→ cosine decay（下降）
        # total_steps の算出にはトレーナーの情報が必要
        total_steps = self.trainer.estimated_stepping_batches
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=lr,
                total_steps=total_steps,
                pct_start=self.opt_conf.lr_pct_start,
                div_factor=self.opt_conf.lr_div_factor,
                final_div_factor=self.opt_conf.lr_final_div_factor,
                anneal_strategy="cos",
            ),
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": opt, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx, **kwargs):
        out = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]

        self.log(
            "loss_train_onset",
            out["onset_loss"],
            batch_size=batch_size,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "loss_train_va",
            out["vad_loss"],
            batch_size=batch_size,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "loss_train_filler",
            out["filler_loss"],
            batch_size=batch_size,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        filler_coef = 0.1 if getattr(self.conf, "use_filler", True) else 0.0
        loss = out["onset_loss"] + out["contrast_loss"] + out["diff_loss"] + out["vad_loss"] + filler_coef * out["filler_loss"]

        # Log batch loss
        self.log(
            "batch_loss",
            loss,
            batch_size=batch_size,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "learning_rate", current_lr, on_step=True, on_epoch=False, sync_dist=True
        )

        # Display training loss info
        if batch_idx % 100 == 0:
            logger.info(
                f"Epoch {self.current_epoch}, Batch {batch_idx}: "
                f"Train Loss = {loss:.4f} (Onset: {out['onset_loss']:.4f}, Ctrst: {out['contrast_loss']:.4f}, Diff: {out['diff_loss']:.4f}, VAD: {out['vad_loss']:.4f}, Filler: {out['filler_loss']:.4f}), "
                f"LR = {current_lr:.6f}"
            )

        return {"loss": loss}

    def on_before_optimizer_step(self, optimizer):
        # Log gradient norms
        total_norm = 0
        param_count = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        if param_count > 0:
            total_norm = total_norm ** (1.0 / 2)
            self.log(
                "gradient_norm",
                total_norm,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )

    def validation_step(self, batch, batch_idx, **kwargs):
        """validation step"""
        if not hasattr(self, "val_metrics"):
            self.val_metrics = self.get_metrics()

        out = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]

        self.log(
            "val_loss_onset",
            out["onset_loss"],
            batch_size=batch_size,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "val_loss_va",
            out["vad_loss"],
            batch_size=batch_size,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )
        self.log(
            "val_loss_filler",
            out["filler_loss"],
            batch_size=batch_size,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
        )

        self.log(
            "val_loss_contrast",
            out["contrast_loss"],
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "val_loss_diff",
            out["diff_loss"],
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        # 合計
        _fc = 0.1 if getattr(self.conf, "use_filler", True) else 0.0
        val_total = out["onset_loss"] + out["contrast_loss"] + out["diff_loss"] + out["vad_loss"] + _fc * out["filler_loss"]
        self.log(
            "val_loss_total",
            val_total,
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        # Event Metrics
        if self.event_extractor is not None:
            events = self.event_extractor(batch["va"])
            preds, targets = self.extract_predictions(
                out["onset_proximity"], events
            )
            self.metrics_step(preds, targets, split="val")

        # フィラー Top-K 正解率の更新
        self._update_filler_topk_metrics(out, batch, split="val")



    def on_validation_epoch_end(self, *_):
        if hasattr(self, "val_metrics"):
            val_loss_onset = self.trainer.callback_metrics.get("val_loss_onset", 0.0)
            val_loss_va = self.trainer.callback_metrics.get("val_loss_va", 0.0)
            val_loss_filler = self.trainer.callback_metrics.get("val_loss_filler", 0.0)
            val_loss_total = self.trainer.callback_metrics.get("val_loss_total", 0.0)

            logger.info(f"Validation Epoch {self.current_epoch} completed:")
            logger.info(
                f"  Val Loss: Total={val_loss_total:.4f} (Onset={val_loss_onset:.4f}, Ctrst={self.trainer.callback_metrics.get('val_loss_contrast', 0.0):.4f}, Diff={self.trainer.callback_metrics.get('val_loss_diff', 0.0):.4f}, VAD={val_loss_va:.4f}, Filler={val_loss_filler:.4f})"
            )

            self.metrics_epoch("val")

    def test_step(self, batch, batch_idx, **kwargs):
        """validation step"""
        if not hasattr(self, "test_metrics"):
            self.test_metrics = self.get_metrics()

            for name, events in self.test_metrics.items():
                for event, metric in events.items():
                    strname = f"test_{name}_{event}"
                    self.register_module(strname, metric)

        # if not hasattr(self, "test_sh_metrics_conf"):
        #     self.test_sh_metrics_conf = MulticlassConfusionMatrix(2)

        #
        # Perterbation
        #

        # import torchaudio
        # torchaudio.save("flatintensity_sample-before.wav", batch["waveform"][0, :, :].cpu(), 16000)

        # #pert = FlatIntensity(min_intensity=20)
        # if self.test_perturbation == 1:
        #     pert = FlatPitch()
        #     batch["waveform"] = pert(batch["waveform"])

        # if self.test_perturbation == 2:

        #     # import torchaudio
        #     # torchaudio.save("lowpass_sample-before.wav", batch["waveform"][0, :, :].cpu(), 16000)

        #     pert = LowPass()
        #     batch["waveform"] = pert(batch["waveform"])

        #     # # import torchaudio
        #     # torchaudio.save("lowpass_sample-after.wav", batch["waveform"][0, :, :].cpu(), 16000)

        # pert = FlatPitch()
        # batch["waveform"] = pert(batch["waveform"])

        # # torchaudio.save("flatintensity_sample-after.wav", batch["waveform"][0, :, :].cpu(), 16000)
        # # input()

        out = self.shared_step(batch)
        batch_size = batch["waveform"].shape[0]

        self.log("test_loss_onset", out["onset_loss"], batch_size=batch_size, sync_dist=True)
        self.log("test_loss_va", out["vad_loss"], batch_size=batch_size, sync_dist=True)
        self.log("test_loss_filler", out["filler_loss"], batch_size=batch_size, sync_dist=True)

        # Event Metrics
        if self.event_extractor is not None:
            events = self.event_extractor(batch["va"])
            preds, targets = self.extract_predictions(
                out["onset_proximity"], events
            )

            # Hold/Shift
            if "hs" in targets:
                if targets["hs"] is not None:
                    for r in targets["hs"]:
                        self.test_hs_reference[r] += 1

                    for p, t in zip(preds["hs"], targets["hs"]):
                        if torch.isnan(p) or torch.isnan(t):
                            continue
                        p_ = torch.round(p)
                        label = "ref=%d:pre=%d" % (t, p_)
                        self.test_hs_confusion_matrix[label] += 1

            # Shift prediction
            if "pred_shift" in targets:
                if targets["pred_shift"] is not None:
                    for r in targets["pred_shift"]:
                        self.test_pred_shift_reference[r] += 1

                    for p, t in zip(preds["pred_shift"], targets["pred_shift"]):
                        if torch.isnan(p) or torch.isnan(t):
                            continue
                        p_ = torch.round(p)
                        label = "ref=%d:pre=%d" % (t, p_)
                        self.test_pred_shift_confusion_matrix[label] += 1

            # Backchannel
            if "bc" in targets:
                if targets["bc"] is not None:
                    for r in targets["bc"]:
                        self.test_bc_reference[r] += 1

                    for p, t in zip(preds["bc"], targets["bc"]):
                        if torch.isnan(p) or torch.isnan(t):
                            continue
                        p_ = torch.round(p)
                        label = "ref=%d:pre=%d" % (t, p_)
                        self.test_bc_confusion_matrix[label] += 1

            self.metrics_step(preds, targets, split="test")

        # フィラー Top-K 正解率の更新
        self._update_filler_topk_metrics(out, batch, split="test")



    def on_test_epoch_end(self, *_):
        if hasattr(self, "test_metrics"):
            self.metrics_epoch("test")


if __name__ == "__main__":
    train()
