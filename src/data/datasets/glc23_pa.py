import numpy as np
import pandas as pd
import torch

from src.data.datasets.abstract_dataset import AbstractDataset


class GLC23PADataset(AbstractDataset):
    def __init__(self, predictors, dataset_file_path):
        super().__init__(predictors)
        self.data = pd.read_csv(dataset_file_path, sep=";", header="infer", low_memory=False)
        self.data = (
            self.data.groupby(["patchID", "dayOfYear", "lat", "lon"])
            .agg({"speciesId": lambda x: list(x)})
            .reset_index()
        )
        self.predictors = predictors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data.iloc[idx]
        lon, lat = tuple(data_dict[["lon", "lat"]].to_numpy())
        sample = self.sample_location(lon, lat, None)
        y = torch.zeros(10040)
        y[data_dict["speciesId"]] = 1
        sample["y"] = y
        return sample
