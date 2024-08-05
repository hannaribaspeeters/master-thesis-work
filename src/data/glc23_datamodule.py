from typing import Any, Dict, Optional, Tuple

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from src.data.abstract_datamodule import AbstractDataModule
from src.data.datasets.glc23_pa import GLC23PADataset
from src.data.datasets.glc24_pa import GLC24PADataset
from src.data.datasets.glc23_pa_predict import GLC23PAPredictDataset
from src.data.datasets.glc_po import GLCPODataset
from src.data.datasets.pseudo_absence_datasets.random_location_psab_ds import (
    RandomLocationDataset,
)
from src.data.datasets.pseudo_absence_datasets.sample_wrapper import (
    PseudoAbsenceSampler,
)
from src.utils.grid_thinning.thinning_strategies import (
    thin_all_species,
    thin_majority_species,
    thin_majority_minority_species,
    thin_bioclim_all_species
)
from time import time

class GLC23DataModule(AbstractDataModule):
    """`LightningDataModule` for the GLC-PO dataset."""

    def __init__(
        self,
        data_path: str = "data/",
        file_path: str = "",
        predictors = None,
        batch_size: int = 2048,
        num_workers: int = 0,
        pin_memory: bool = False, 
        pseudo_absence_num_saved_batches: int = 256,
        pseudo_absence_sampling_bounds = Dict[str, int],
        species_count_threshold: int = 0,
        spatial_thinning = None,
        majority_cutoff = 100,
        thin_dist: float = 1,
        weighted_random_sampler = None,
        input_noise = None, 
        eps = 0,
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

        self.data_test = GLC23PADataset(
            self.hparams.predictors,
            f"{self.hparams.data_path}Presence_Absence_surveys/Presences_Absences_train-test.csv",
        )

        self.data_val = GLC23PADataset(
            self.hparams.predictors,
            f"{self.hparams.data_path}Presence_Absence_surveys/Presences_Absences_train-val.csv",
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
            dataset=pdsb_ds,
            datamodule=self,
            num_saved_batches=pseudo_absence_num_saved_batches
        )

        self.weighted_random_sampler = weighted_random_sampler


    def setup(self, stage: Optional[str] = None) -> None:
        
        super().setup(stage)  # Call setup method of the parent class

        timer = time()

        # Load the training data
        self.data_train = GLCPODataset(
            self.hparams.predictors,
            f"{self.hparams.data_path}processed_po_data/{self.hparams.file_path}",
            input_noise=self.hparams.input_noise,
        )
            
        if stage == "fit":
            
            # THINNING 
            # 3D Spatial thinning
            if self.hparams.spatial_thinning=="thin_all":
                self.data_train.data, _ = thin_all_species(
                    df=self.data_train.data, 
                    thin_dist=self.hparams.thin_dist
                )   
            elif self.hparams.spatial_thinning=="thin_majority":
                self.data_train.data, _ = thin_majority_species(
                    df=self.data_train.data, 
                    thin_dist=self.hparams.thin_dist,
                    majority_cutoff=self.hparams.majority_cutoff
                )
            # 20D Bioclimatic thinning
            elif self.hparams.spatial_thinning=="thin_bioclimatic":
                self.data_train.data, _ = thin_bioclim_all_species(
                    dataset=self.data_train,
                    thin_dist=self.hparams.thin_dist,
                    save=True
                )
            elif self.hparams.spatial_thinning is None: 
                pass
            else:
                raise ValueError(
                    "Invalid value for 'spatial_thinning'. Expected one of\
                    ['thin_all', 'thin_majority', 'thin_bioclimatic', None]."
                    ) 
            
            species_counts = self.data_train.data["speciesId"].value_counts()
            
            if self.hparams.species_count_threshold>0:
                # Filter the training data by removing instances of species
                # with less than a certain threshold of occurrences
                filtered_species = species_counts[
                    species_counts >= self.hparams.species_count_threshold
                ].index.tolist()
                self.data_train.data = self.data_train.data[
                    self.data_train.data["speciesId"].isin(filtered_species)
                ].copy()
                            
            print(f"Completed setting up the data in \
                  {np.round(time()-timer, 4)} s.")

            # CREATE A WEIGHTED RANDOM SAMPLER FOR TRAINING

            if self.weighted_random_sampler == "inverse_counts":
                # Each sample's weight is the inverse of its species' count
                sample_weights = [
                    1/(np.sqrt(species_counts[i]) + self.hparams.eps)\
                    for i in self.data_train.data["speciesId"].values
                ]
                
                self.weighted_random_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(self.data_train.data),
                replacement=True
                )
                
            elif self.weighted_random_sampler == "inverse_cluster_density":
                # Each sample's weight is the inverse of its species count
                # after thinning de data divided by the density of the sample's
                # cluster

                data = self.data_train.data
                sampled_data, cluster_density = thin_all_species(
                    data, 
                    thin_dist=self.hparams.thin_dist
                )
                sampled_species_counts = sampled_data["speciesId"].\
                    value_counts()
                class_weights =  [
                    1/(np.sqrt(sampled_species_counts[i]*species_counts[i]) +\
                    self.hparams.eps) for i in self.data_train.\
                    data["speciesId"].values
                ]
                assert(len(data)==len(cluster_density))
                assert(len(set(data.index)-set(cluster_density.index))==0)

                # the merge is done to ensure that the right cluster
                # densities are assigned to the right rows 
                data = data.merge(
                    cluster_density.rename("cluster_density"),
                    left_index=True,
                    right_index=True
                )
                cluster_density_values = data["cluster_density"].values
                assert(len(class_weights)==len(cluster_density_values))

                sample_weights = class_weights/cluster_density_values

                self.weighted_random_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(self.data_train.data),
                replacement=True
                )
            
            elif self.weighted_random_sampler is None: 
                pass

            else:
                raise ValueError(
                    "Invalid value for 'weighted_random_sampler'.Expected one\
                    of ['inverse_counts', 'inverse_cluster_density', None]."
                    )

        self.data_train.species_counts = self.data_train.data["speciesId"]\
        .value_counts()
        self.data_train.n_locations = len(
            self.data_train.data.drop_duplicates(subset=["lon", "lat"])
            )
        
    # Overriding the train_dataloader method to incorporate the sampler 
    def train_dataloader(self):
        # Determine whether to reshuffle the data at every epoch based
        # on whether a sampling strategy is defined
        shuffle = self.weighted_random_sampler is None
        return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size_per_device,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=shuffle,
                sampler=self.weighted_random_sampler
            )

    @property
    def num_classes(self) -> int:
        """Get the number of classes.
        :return: The number of classes (10040).
        """
        return 10040

if __name__ == "__main__":
    _ = GLC23DataModule()
