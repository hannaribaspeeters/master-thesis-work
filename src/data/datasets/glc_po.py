import pandas as pd
import torch

from src.data.datasets.abstract_dataset import AbstractDataset

class GLCPODataset(AbstractDataset):
    def __init__(
            self,
            predictors,
            dataset_file_path,
            input_noise=None,
            input_noise_var=False
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
        self.input_noise_var = input_noise_var
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

        # The noise distribution varies from one variable to another
        # In particular, the standard deviation is calculated as a fraction
        # of the max-min range of that variable         
        if self.input_noise_var:
            path = "/shares/wegner.ics.uzh/glc23_data/bioclim+elev/bioclim_var_range.pt"
            variable_range = torch.load(path)
            noise_var = variable_range*self.input_noise
            noise = torch.randn_like(sample["bioclim_pointwise_europe"])*\
                noise_var
            sample["bioclim_pointwise_europe"] += noise

        # the noise distribution is the same for all variables
        elif self.input_noise is not None:
            noise = torch.randn_like(sample["bioclim_pointwise_europe"])*\
                self.input_noise
            sample["bioclim_pointwise_europe"] += noise

        return sample
