from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from src.data.abstract_datamodule import AbstractDataModule
from src.data.datasets.glc_po import GLCPODataset


class GLCPODataModule(AbstractDataModule):
    """`LightningDataModule` for the GLC-PO dataset."""

    def __init__(
        self,
        dataset_file_path: str = "data/",
        predictors=None,
        train_val_test_split: Tuple[int, int, int] = (0.85, 0.05, 0.1),
        batch_size: int = 2048,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `GLCPODataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(0.85, 0.05, 0.1)`.
        :param batch_size: The batch size. Defaults to `2048`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__(
            predictors,
            batch_size,
            num_workers,
            pin_memory,
        )

        dataset = GLCPODataset(self.hparams.predictors, self.hparams.dataset_file_path)
        self.data_train, self.data_val, self.data_test = random_split(
            dataset=dataset,
            lengths=self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(42),
        )

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of classes (10040).
        """
        return 10040


if __name__ == "__main__":
    _ = GLCPODataModule()
