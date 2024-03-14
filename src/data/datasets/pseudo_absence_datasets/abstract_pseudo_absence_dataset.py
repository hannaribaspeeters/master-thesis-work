from abc import ABC
from typing import Dict

from src.data.abstract_datamodule import AbstractDataModule
from src.data.datasets.abstract_dataset import AbstractDataset
from src.data.predictors.abstract_predictor import AbstractPredictor


class AbstractPseudoAbsenceDataset(ABC, AbstractDataset):
    """Abstract class for pseudo-absence datasets."""

    def __init__(self, datamodule: AbstractDataModule, num_saved_batches: int):
        super().__init__(datamodule.hparams.predictors)
        self.datamodule = datamodule
        self.batch_size = datamodule.hparams.batch_size
        self.num_saved_batches = num_saved_batches

    def __len__(self):
        """Tells the dataloader how much space should be requested, and how often the dataset can
        be sampled before restocking."""
        # We want to always provide as many batches as possible to increase training speed, but for each batch storage gets allocated
        # This is a tradeoff between speed and allocated storage
        return self.batch_size * self.num_saved_batches
