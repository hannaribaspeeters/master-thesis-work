import numpy as np
import torch

from abc import ABC

class AbstractPredictor(ABC):
    def __init__(self, bounds):
        self.bounds = bounds

    def sample_location(self, lon: float, lat: float, time: str):
        raise NotImplementedError()

    def __str__(self):
        # This is supposed to print meta-information about the predictor and its bounds
        raise NotImplementedError()

    def check_bounds(self, lon, lat) -> bool:
        return (self.bounds["north"] > lat and self.bounds["south"] < lat and self.bounds["east"] > lon and self.bounds["west"] < lon )
