from abc import ABC
from typing import Dict

import torch

from src.data.abstract_datamodule import AbstractDataModule


class AbstractPASampler(ABC):
    """Abstract class to define interface with the sampler."""

    def __init__(self, datamodule: AbstractDataModule) -> None:
        """
        :params datamodule: Giving Sampler access to the datamodule for predictors and informed sampling.
        """
        self.datamodule = datamodule

    def get_samples(self) -> Dict[str, torch.tensor]:
        """Get PA samples.

        Amount is defined by the sub-classes.
        """
        raise NotImplementedError()
