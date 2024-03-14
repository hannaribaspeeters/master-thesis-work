import warnings
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.data.abstract_datamodule import AbstractDataModule
from src.data.predictors.abstract_predictor import AbstractPredictor
from src.data.pseudo_absence_samplers.abstract_pseudo_absence_sampler import (
    AbstractPASampler,
)


class _RandomLocationDataset(torch.torch.utils.data.Dataset):
    """Dataset, that returns random batches."""

    def __init__(self, bounds, predictors, batch_size, num_saved_batches):
        """Dataset, that returns random batches.

        :param bounds: Defines the bounds from which the loss should sample the background samples.
        :param predictors: predictors that should be passed for every iteration.
        """
        self.bounds = bounds
        self.predictors = predictors
        self.batch_size = batch_size
        self.num_saved_batches = num_saved_batches

    def sample_location(self, lon, lat, time=None):
        """Taken from dataset.

        Wrapper function fusing all predictors into a single dict while being oblivious of their
        content.
        """
        sample = dict()
        for name, pred in self.predictors.items():
            sample[name] = pred.sample_location(lon, lat, time)
        return sample

    def __len__(self):
        """Tells the dataloader how much space should be requested, and how often the dataset can
        be sampled before restocking."""
        # We want to always provide as many batches as possible to increase training speed, but for each batch storage gets allocated
        # This is a tradeoff between speed and allocated storage
        return self.batch_size * self.num_saved_batches

    def __getitem__(self, idx):
        """Returns a randomly sampled embedding from within the bounds."""
        lat = (
            float(torch.rand(1)) * (self.bounds["north"] - self.bounds["south"])
            + self.bounds["south"]
        )
        lon = (
            float(torch.rand(1)) * (self.bounds["east"] - self.bounds["west"])
            + self.bounds["west"]
        )
        return self.sample_location(lon, lat)


class RandomLocationSampler(AbstractPASampler):
    """Wrapper around the dataloader of the _RandomLocationDataset.

    Here to take care of restocking the dataloader if empty.
    """

    def __init__(
        self, bounds: Dict[str, int], datamodule: AbstractDataModule, num_saved_batches: int
    ):
        super().__init__(datamodule)
        rld = _RandomLocationDataset(
            bounds, datamodule.hparams.predictors, datamodule.hparams.batch_size, num_saved_batches
        )
        self.rld_dl = DataLoader(
            rld,
            batch_size=datamodule.hparams.batch_size,
            num_workers=datamodule.hparams.num_workers,
            pin_memory=datamodule.hparams.pin_memory,
        )
        self.rld_dl_iter = iter(self.rld_dl)
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
        samples = next(self.rld_dl_iter)
        self.batch_counter += 1
        if self.batch_counter == self.num_saved_batches:
            self.rld_dl_iter = iter(self.rld_dl)
        return samples
