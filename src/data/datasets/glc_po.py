import pandas as pd
import torch

from src.data.datasets.abstract_dataset import AbstractDataset

class GLCPODataset(AbstractDataset):
    def __init__(
            self,
            predictors,
            dataset_file_path,
            input_noise=None,
        ):
        super().__init__(predictors)
        self.data = pd.read_csv(
            dataset_file_path,
            sep=";",
            header="infer",
            low_memory=False
        )
        self.predictors = predictors
        self.input_noise = input_noise
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data.iloc[idx]
        lon, lat = tuple(data_dict[["lon", "lat"]].to_numpy())
        # enrich the sample with the predictors
        sample = self.sample_location(lon, lat, None)
        y = torch.zeros(10040)
        y[data_dict["speciesId"].astype(int)] = 1
        sample["y"] = y

        # Adding gaussian noise to bioclimatic variables
        # the noise distribution is the same for all variables
        if self.input_noise is not None:
            noise = torch.randn_like(sample["bioclim_pointwise_europe"])*\
                self.input_noise
            sample["bioclim_pointwise_europe"] += noise

        return sample
