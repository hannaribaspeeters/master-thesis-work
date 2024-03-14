from abc import ABC

import torch
from lightning import LightningModule

from src.data.abstract_datamodule import AbstractDataModule


class AbstractCriterion(ABC):
    """Abstract class for all criterions."""

    def __init__(self):
        pass

    def __call__(self, logits: torch.Tensor, y: torch.Tensor):
        """Gets called by the model.

        :params logits: Output of the network without softmax/sigmoid.
        :params y: binary torch.Tensor of size [batch_size, num_classes].
        """
        raise NotImplementedError()

    def set_model(self, model: LightningModule):
        """Abstract function to let loss have access to the model for background sampling."""
        pass

    def set_datamodule(self, datamodule: AbstractDataModule):
        """Abstract function to let loss have access to the dataset/predictors for background
        sampling."""
        pass
