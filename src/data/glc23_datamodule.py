from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, WeightedRandomSampler

from src.data.abstract_datamodule import AbstractDataModule
from src.data.datasets.glc23_pa import GLC23PADataset
from src.data.datasets.glc23_pa_predict import GLC23PAPredictDataset
from src.data.datasets.glc_po import GLCPODataset
from src.data.datasets.pseudo_absence_datasets.random_location_psab_ds import (
    RandomLocationDataset,
)
from src.data.datasets.pseudo_absence_datasets.sample_wrapper import (
    PseudoAbsenceSampler,
)

class GLC23DataModule(AbstractDataModule):
    """`LightningDataModule` for the GLC-PO dataset."""

    def __init__(
        self,
        data_path: str = "data/",
        file_path: str = "",
        predictors=None,
        batch_size: int = 2048,
        num_workers: int = 0,
        pin_memory: bool = False, 
        pseudo_absence_num_saved_batches: int = 256,
        pseudo_absence_sampling_bounds=Dict[str, int],
        sampler=None, 
        species_count_threshold: int=0,
    ) -> None:
        """Initialize a `GLC23DataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `2048`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        :param sampler: Defines the strategy to draw samples from the dataset. Defaults to None. 
        :param species_count_threshold: Minimum occurrences of species required
        for inclusion in the dataset. Defaults to 0
        """
        super().__init__(
            predictors,
            batch_size,
            num_workers,
            pin_memory,
        )

        self.sampler = sampler

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

        pdsb_ds = RandomLocationDataset(
            bounds=pseudo_absence_sampling_bounds,
            datamodule=self,
            num_saved_batches=pseudo_absence_num_saved_batches,
        )
        self.pseudo_absence_sampler = PseudoAbsenceSampler(
            dataset=pdsb_ds, datamodule=self, num_saved_batches=pseudo_absence_num_saved_batches
        )

    def setup(self, stage: Optional[str] = None) -> None:
        
        super().setup(stage)  # Call setup method of the parent class

        # Load the training data
        self.data_train = GLCPODataset(
            self.hparams.predictors,
            f"{self.hparams.data_path}{self.hparams.file_path}"
        )
        
        # Filter the training data by removing instances of species with less than a certain threshold of occurrences
        species_counts = self.data_train.data["speciesId"].value_counts()
        filtered_species = species_counts[
            species_counts >= self.hparams.species_count_threshold
            ].index.tolist()
        
        self.data_train.data = self.data_train.data[
            self.data_train.data["speciesId"].isin(filtered_species)
            ]

        # Calculate and store the count of each species and the number of unique locations in the training data
        self.data_train.species_counts = species_counts
        self.data_train.n_locations = len(
            self.data_train.data.drop_duplicates(subset=["lon", "lat"])
            )

        if self.sampler == "WeightedRandomSampler":
            # Create a weighted random sampler for training, where each sample's weight is the inverse of its species' count
            sample_weights = [
                1/species_counts[i] for i in\
                self.data_train.data["speciesId"].values
                ]
            self.sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(self.data_train.data),
                replacement=True
            )
    # Overriding the train_dataloader method to incorporate the sampler parameter
    def train_dataloader(self):
        # Determine whether to reshuffle the data at every epoch based on whether a sampling strategy is defined
        shuffle = self.sampler is None
        return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=shuffle,
                sampler=self.sampler
            )
    
    @property
    def num_classes(self) -> int:
        """Get the number of classes.
        :return: The number of classes (10040).
        """
        return 10040

if __name__ == "__main__":
    _ = GLC23DataModule()
