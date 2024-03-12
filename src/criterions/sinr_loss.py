import warnings
from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.criterions.abstract_criterion import AbstractCriterion


class RandomLocationSampler(torch.torch.utils.data.Dataset):
    """Dataset, that returns random batches."""

    def __init__(self, bounds, predictors, batch_size, num_saved_batches):
        """Dataset, that returns random batches.

        :param bounds: Defines the bounds from which the loss should sample the background samples.
        :parama predictors: predictors that should be passed for every iteration.
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


class SinrANFull(AbstractCriterion):
    """The full "assume negative" loss from SINR."""

    def __init__(
        self,
        pos_weight: int = 2048,
        bounds: Dict[str, int] = {"north": 90, "south": -90, "west": -180, "east": 180},
        batch_size: int = 2048,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_saved_batches: int = 16,
    ) -> None:
        """Implementation of the full "assume negative" loss from SINR.

        :param pos_weight: Up-scaling of the label prediction.
        :param bounds: Defines the bounds from which the loss should sample the background samples.
        :param batchsize: Defines how many samples should be taken per batch.
        :param num_workers: How many workers the sampler should use.
        :param pin_memory: Torch dataloader arg.
        :param num_saved_batches: Amount of batches per full dataloader.
        """
        super().__init__()
        self.bounds = bounds
        self.pos_weight = pos_weight
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_saved_batches = num_saved_batches

        if num_saved_batches < num_workers:
            warnings.warn(
                "You are saving less batches than number of workers, possibly increasing runtime."
            )

        self.sampler: RandomLocationSampler = None

    def set_dataset(self, dataset):
        """Overriding this call to instantiate the sampler dataloader for background sampling."""
        if isinstance(dataset, torch.utils.data.Subset):
            sampler = RandomLocationSampler(
                self.bounds, dataset.dataset.predictors, self.batch_size, self.num_saved_batches
            )
        else:
            sampler = RandomLocationSampler(
                self.bounds, dataset.predictors, self.batch_size, self.num_saved_batches
            )
        self.sample_loader = DataLoader(
            dataset=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        self.sample_iterator = iter(self.sample_loader)
        self.batch_id = 0

    def set_model(self, model):
        """For access to the model.

        Called by the main train function.
        """
        self.model = model

    def __call__(self, logits, y) -> torch.Tensor:
        """Sampling random background samples, then calculating final loss."""

        # Sample a batch of random locations
        try:
            random_samples = next(self.sample_iterator)
        except RuntimeError:  # Necessary for graceful quitting after Control+C
            pass
        self.batch_id += 1
        if self.batch_id == self.num_saved_batches:
            # Have to reset the iterator every #num_saved_batches batches
            # If you notice your training stopping every #num_saved_batches steps for a few seconds, then here is the reason why
            self.sample_iterator = iter(self.sample_loader)
            self.batch_id = 0

        for key, val in random_samples.items():
            random_samples[key] = val.to(y.device)  # Put the random samples on the proper device
        random_loc_logits = self.model(random_samples)  # Get their logits from the model

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
