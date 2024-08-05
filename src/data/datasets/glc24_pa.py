import numpy as np
import pandas as pd
import torch

from src.data.datasets.abstract_dataset import AbstractDataset
from src.data.predictors.abstract_predictor import AbstractPredictor


class GLC24PADataset(AbstractDataset):
    def __init__(self, predictors, path):
        super().__init__(predictors)
        self.data = pd.read_csv(path, header="infer", low_memory=False)
        self.data["speciesId"] = self.data["speciesId"].astype(int)
        self.data = (
            self.data.groupby(["surveyId", "year", "lat", "lon"])
            .agg({"speciesId": lambda x: list(x)})
            .reset_index()
        )
        self.predictors=predictors

    @property
    def num_classes(self):
        """Return the number of classes."""
        return 11255

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data.iloc[idx]
        lon, lat = tuple(data_dict[["lon", "lat"]].to_numpy())
        sample = self.sample_location(lon, lat, None)
        y = torch.zeros(10040)
        y[np.clip(data_dict["speciesId"], 0, 10039)] = 1
        sample["y"] = y
        return sample