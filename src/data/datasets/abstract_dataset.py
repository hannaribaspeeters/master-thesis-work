import torch

class AbstractDataset(torch.utils.data.Dataset):
    def __init__(self, predictors):
        self.predictors = predictors

    def sample_location(self, lon, lat, time):
        sample = dict()
        for name, pred in self.predictors.items():
            sample[name] = pred.sample_location(lon, lat, time)
        return sample

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()
