from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from src.data.abstract_datamodule import AbstractDataModule
from src.data.datasets.glc23_pa import GLC23PADataset
from src.data.datasets.glc23_pa_predict import GLC23PAPredictDataset
from src.data.datasets.glc_po import GLCPODataset


class GLC23DataModule(AbstractDataModule):
    """`LightningDataModule` for the GLC-PO dataset."""

    def __init__(
        self,
        data_path: str = "data/",
        predictors=None,
        train_val_test_split: Tuple[int, int, int] = (0.85, 0.05, 0.1),
        batch_size: int = 2048,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `GLC23DataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(0.85, 0.05, 0.1)`.
        :param batch_size: The batch size. Defaults to `2048`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__(
            data_path,
            predictors,
            train_val_test_split,
            batch_size,
            num_workers,
            pin_memory,
        )

        self.data_train = GLCPODataset(
            self.hparams.predictors, f"{self.hparams.data_path}Presences_only_train.csv"
        )
        self.data_val = GLC23PADataset(
            self.hparams.predictors,
            f"{self.hparams.data_path}Presence_Absence_surveys/Presences_Absences_train.csv",
        )
        self.data_test = GLC23PADataset(
            self.hparams.predictors,
            f"{self.hparams.data_path}Presence_Absence_surveys/Presences_Absences_train.csv",
        )
        self.data_predict = GLC23PAPredictDataset(
            self.hparams.predictors, f"{self.hparams.data_path}For_submission/test_blind.csv"
        )

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of classes (10040).
        """
        return 10040


if __name__ == "__main__":
    _ = GLC23DataModule()
