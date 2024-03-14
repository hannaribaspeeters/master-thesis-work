import warnings
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.data.abstract_datamodule import AbstractDataModule
from src.data.datasets.pseudo_absence_datasets.abstract_psab_dataset import (
    AbstractPsAbDataset,
)


class PseudoAbsenceSampler:
    """Wrapper around the dataloader of the pseudo absence dataset.

    Here to take care of restocking the dataloader if empty.
    """

    def __init__(
        self, dataset: AbstractPsAbDataset, datamodule: AbstractDataModule, num_saved_batches: int
    ):
        """
        :param dataset: Dataset that is being wrapped.
        :param datamodule: Datamodule containing the hparams.
        :param num_saved_batches: Number of batches after which the dataloader needs restocking."""
        self.dataloader = DataLoader(
            dataset,
            batch_size=datamodule.hparams.batch_size,
            num_workers=datamodule.hparams.num_workers,
            pin_memory=datamodule.hparams.pin_memory,
        )
        self.dl_iter = iter(self.dataloader)
        self.batch_counter = 0
        self.num_saved_batches = num_saved_batches

        if num_saved_batches < datamodule.hparams.num_workers:
            warnings.warn(
                "You are saving less batches than number of workers, possibly increasing runtime."
            )

    def get_samples(self) -> Dict[str, torch.tensor]:
        """First get a sample from the iterator.

        If it is empty, restock.
        """
        samples = next(self.dl_iter)
        self.batch_counter += 1
        if self.batch_counter == self.num_saved_batches:
            self.dl_iter = iter(self.dataloader)
            self.batch_counter = 0
        return samples
