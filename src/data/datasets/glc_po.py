import pandas as pd
import torch


class GLCPODataset(torch.utils.data.Dataset):
    def __init__(self, predictors, dataset_file_path):
        self.data = pd.read_csv(dataset_file_path, sep=";", header="infer", low_memory=False)
        self.predictors = predictors

    def sample_location(self, lon, lat, time):
        sample = dict()
        for name, pred in self.predictors.items():
            sample[name] = pred.sample_location(lon, lat, time)
        return sample

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
