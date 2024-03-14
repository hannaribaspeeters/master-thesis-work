import torch

from src.data.datasets.pseudo_absence_datasets.abstract_psab_dataset import (
    AbstractPsAbDataset,
)


class RandomLocationDataset(AbstractPsAbDataset):
    """Dataset, that returns random batches."""

    def __init__(self, bounds, datamodule, num_saved_batches):
        """Dataset, that returns random batches from within specified bounds.

        :param bounds: Defines the bounds from which the loss should sample the background samples.
        """
        super().__init__(datamodule, num_saved_batches)
        self.bounds = bounds

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
        return self.sample_location(lon, lat, None)
