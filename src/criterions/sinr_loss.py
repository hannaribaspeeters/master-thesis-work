import warnings
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.criterions.abstract_criterion import AbstractCriterion
from src.data.abstract_datamodule import AbstractDataModule


class SinrANFull(AbstractCriterion):
    """The full "assume negative" loss from SINR."""

    def __init__(
        self,
        pos_weight: int = 2048,
    ) -> None:
        """Implementation of the full "assume negative" loss from SINR.

        :param pos_weight: Up-scaling of the label prediction.
        """
        super().__init__()
        self.pos_weight = pos_weight

    def set_datamodule(self, datamodule: AbstractDataModule):
        """We only require the pseudo_absence_sampler in this loss."""
        self.pseudo_absence_sampler = datamodule.pseudo_absence_sampler

    def set_model(self, model):
        """For access to the model.

        Called by the main train function.
        """
        self.model = model

    def __call__(self, logits, y) -> torch.Tensor:
        """Sampling random background samples, then calculating final loss."""

        # Sample a batch of random locations
        random_samples = self.pseudo_absence_sampler.get_samples()
        for key, val in random_samples.items():
            random_samples[key] = val.to(y.device)  # Put the random samples on the proper device
        random_loc_logits = self.model(random_samples)  # Get their logits from the model

        # Applying sigmoid to the logits to get probabilities
        loc_pred = torch.sigmoid(logits)
        rand_pred = torch.sigmoid(random_loc_logits)[
            : len(loc_pred)
        ]  # cutting off so we never have more random than non-random

        assert len(rand_pred) == len(loc_pred)

        # Assumption, that y is a single/multi-label binary vector of presence/absence
        loss_pos = -torch.log((1 - loc_pred) + 1e-5) * (
            1 - y
        )  # All negative classes at location get penalized
        loss_pos += (
            self.pos_weight * -torch.log(loc_pred + 1e-5) * y
        )  # All positive classes at location get rewarded

        loss_bg = -torch.log(
            (1 - rand_pred) + 1e-5
        )  # All positions at the random locations get penalized

        return loss_pos.mean() + loss_bg.mean()
