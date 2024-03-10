
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
import numpy as np
import pandas as pd
import csv

class PredictionWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
 
        nonzero_indices = torch.nonzero(prediction)

        data_to_append = []
        for idx in batch_indices:
            # Filter nonzero indices for the current index
            idx_nonzero_indices = nonzero_indices[nonzero_indices[:, 0] == idx]
            predictions = " ".join(str(int(j)) for j in idx_nonzero_indices[:, 1])

            data_to_append.append([idx, predictions])

        # Write data to CSV
        with open(f"{self.output_dir}prediction.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            if batch_idx == 0:
                writer.writerow(["Id", "Prediction"])
            writer.writerows(data_to_append)

