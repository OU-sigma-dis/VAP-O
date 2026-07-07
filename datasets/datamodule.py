from os import cpu_count
from os.path import exists
from typing import Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from datasets.dataset import VapDataset


def collate_fn_float32(batch):
    """カスタムcollate関数：すべてのfloat tensorをfloat32に変換"""
    from torch.utils.data.dataloader import default_collate

    # デフォルトのcollateを適用
    batch = default_collate(batch)

    # すべてのfloat tensorをfloat32に変換
    def convert_to_float32(obj):
        if isinstance(obj, torch.Tensor) and obj.dtype == torch.float64:
            return obj.to(torch.float32)
        elif isinstance(obj, dict):
            return {k: convert_to_float32(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(convert_to_float32(item) for item in obj)
        return obj

    return convert_to_float32(batch)


class VapDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        frame_hz: int = 20,
        sample_rate: int = 16000,
        batch_size: int = 16,
        num_workers: int = 16,
        pin_memory: bool = True,
        window_size: float = 20.0,
        stride: float = 20.0,
        val_stride: float = 20.0,
        max_text_tokens: int = 32,
        cpc_feature_dir: str = "",
    ):
        super().__init__()

        # Files
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        # values
        self.sample_rate = sample_rate
        self.frame_hz = frame_hz

        # DataLoder
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers

        # データセットの共通パラメータ
        self._base_dataset_kwargs = {
            "sample_rate": sample_rate,
            "frame_hz": frame_hz,
            "window_size": window_size,
            "max_text_tokens": max_text_tokens,
            "cpc_feature_dir": cpc_feature_dir,
        }
        self.train_stride = stride
        self.val_stride = val_stride

        # データローダーのパラメータ
        self.dataloader_kwargs = {
            "batch_size": batch_size,
            "pin_memory": pin_memory,
            "num_workers": num_workers,
            "persistent_workers": num_workers > 0,
            "prefetch_factor": 2 if num_workers > 0 else None,
        }

    def prepare_data(self):
        if self.train_path is not None:
            assert self.train_path is not None, "Train path is NONE."
            assert exists(self.train_path), f"No TRAIN file found: {self.train_path}"
        if self.val_path is not None:
            assert self.val_path is not None, "Validation path is NONE."
            assert exists(self.val_path), f"No VAL file found: {self.val_path}"

        if self.test_path is not None:
            assert self.test_path is not None, "Test path is NONE."
            assert exists(self.test_path), f"No TEST file found: {self.test_path}"

    def setup(self, stage: Optional[str] = "fit"):
        """Loads the datasets"""

        if stage in (None, "fit"):
            assert self.train_path is not None, "Train path is NONE."
            assert self.val_path is not None, "Validation path is NONE."
            self.train_dset = VapDataset(
                self.train_path, stride=self.train_stride, **self._base_dataset_kwargs
            )
            self.val_dset = VapDataset(
                self.val_path, stride=self.val_stride, **self._base_dataset_kwargs
            )

        if stage in (None, "test"):
            assert self.test_path is not None, "Test path is NONE."
            self.test_dset = VapDataset(
                self.test_path, stride=self.val_stride, **self._base_dataset_kwargs
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn_float32,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn_float32,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn_float32,
            **self.dataloader_kwargs,
        )

    def __repr__(self):
        s = self.__class__.__name__
        s += f"\n\tTrain: {self.train_path}"
        s += f"\n\tVal: {self.val_path}"
        s += f"\n\tTest: {self.test_path}"
        s += f"\n\tSample rate: {self.sample_rate}"
        s += f"\n\tFrame Hz: {self.frame_hz}"
        s += "\nData"
        s += f"\n\tbatch_size: {self.batch_size}"
        s += f"\n\tpin_memory: {self.pin_memory}"
        s += f"\n\tnum_workers: {self.num_workers}"
        return s

    @staticmethod
    def add_data_specific_args(parent_parser):
        """argparse arguments for SoSIModel (based on yaml-config)"""
        parser = parent_parser.add_argument_group("ULMProjection")
        parser.add_argument("--train_path", default=None, type=str)
        parser.add_argument("--val_path", default=None, type=str)
        parser.add_argument("--test_path", default=None, type=str)
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=cpu_count(), type=int)
        return parent_parser


if __name__ == "__main__":
    from tqdm import tqdm

    # 1. クラスをインスタンス化
    data_manager = VapDataModule(
        train_path="../data/switchboard/vap_dataset/train.csv",
        val_path="../data/switchboard/vap_dataset/val.csv",
        batch_size=16,
        num_workers=16,
        pin_memory=False,
    )

    # 2. データセットを準備
    data_manager.setup("fit")

    print(data_manager)
    print("Train dataset size: ", len(data_manager.train_dset))
    print("Validation dataset size: ", len(data_manager.val_dset))

    # 3. データローダーを取得
    dloader = data_manager.train_dataloader()

    for batch in tqdm(dloader, total=len(dloader)):
        pass
