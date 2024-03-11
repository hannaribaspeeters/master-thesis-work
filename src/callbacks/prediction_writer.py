
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
import csv

class PredictionWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
 
        data_to_append = []
        for i, idx in enumerate(batch_indices):
            
            # get top k indices and format correctly
            top_k_idx = torch.sort(torch.topk(prediction[i], 30, sorted = False)[1])[0]
            predictions = " ".join(str(int(j)) for j in top_k_idx)
            
            data_to_append.append([idx+1, predictions])

        # write data to CSV
        with open(f"{self.output_dir}/prediction.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            if batch_idx == 0:
                writer.writerow(["surveyId", "predictions"])
            writer.writerows(data_to_append)

