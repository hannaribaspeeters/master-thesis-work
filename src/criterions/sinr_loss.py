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
    
class SinrANFull_weighted(AbstractCriterion):
    """The full "assume negative" loss from SINR."""

    def __init__(
        self,
        pos_weight: int = 2048,
        #bg_weight: float = 0.5, 
        freq_to_weight_method=None
    ) -> None:
        """Implementation of the full "assume negative" loss from SINR.

        :param pos_weight: Up-scaling of the label prediction.
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.freq_to_weight_method = freq_to_weight_method
        #self.bg_weight = bg_weight

    def set_datamodule(self, datamodule: AbstractDataModule):
        """We only require the pseudo_absence_sampler in this loss."""
        # Load calculated quantities from the training data such as species_counts and n_locations
        datamodule.setup()
        
        # Obtain the count of each species in the training data, sorted by species ID
        species_counts = datamodule.data_train.species_counts.sort_index()

        # Get the total number of unique locations in the training data
        n_locations = datamodule.data_train.n_locations

        # Initialize a tensor to hold the species counts, with a size equal to the total number of species
        species_counts_tensor = torch.zeros(10040)

        # Fill the tensor with the species counts, using the species IDs as indices
        species_counts_tensor[species_counts.index] = torch.tensor(species_counts.values, dtype=torch.float32)
        
        # Calculate the frequency of each species as the ratio of the number of locations where it's observed to the total number of locations
        self.species_freq = torch.div(species_counts_tensor, n_locations)

        # Calculate the relative frequency of each species as the ratio of its count to the count of the most common species, 
        # while ensuring the values are between 0.01 and 0.99 to avoid division by zero in subsequent calculations
        self.species_rel_freq = torch.div(
            species_counts_tensor,
            species_counts.max()
            ).clamp(0.01, 0.99)
        
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
        # Scale the frequencies to prevent weights from becoming too large
        if self.freq_to_weight_method=="scaled":
            species_freq = self.species_freq
            min_freq, max_freq = species_freq.min(), species_freq.max()
        
            min_scaled_freq, max_scaled_freq = 0.25, 0.75
            # Compute scaling factor
            scale_factor = (max_scaled_freq - min_scaled_freq) / (max_freq - min_freq)
            scaled_freqs = ((species_freq - min_freq)*scale_factor + min_scaled_freq)\
            .to(loc_pred.device)

            w1 = 1/scaled_freqs 
            w2 = 1/(1-scaled_freqs)

        # Compute the species frequency as the relative frequency of each species compared to the most common species
        elif self.freq_to_weight_method=="relative":
            rel_freqs = self.species_rel_freq.to(loc_pred.device)
            w1 = 1/rel_freqs
            w2 = 1/(1-rel_freqs)

        # Only weight the positive loss term
        elif self.freq_to_weight_method=="positive_loss_weighted":
            rel_freqs = self.species_rel_freq.to(loc_pred.device)
            w1 = 1/rel_freqs
            w2 = 1

        # If no frequency-to-weight method is specified, apply no weighting to the loss
        elif self.freq_to_weight_method is None:
            w1 = 1
            w2 = 1

        else:
            raise ValueError("Invalid value for 'freq_to_weight_method'. Expected one of ['scaled', 'relative', 'positive', None].")
        
        # Assumption, that y is a single/multi-label binary vector of presence/absence
        loss_pos = w2*\
            torch.log((1 - loc_pred) + 1e-5)*(1 - y)
    
        # All negative classes at location get penalized
        loss_pos += (
                self.pos_weight *\
                w1*\
                torch.log(loc_pred + 1e-5) * y
        )  # All positive classes at location get rewarded
        loss_bg = torch.log(
            (1 - rand_pred) + 1e-5
        )  # All positions at the random locations get penalized

        return - (loss_pos.mean() + loss_bg.mean())
