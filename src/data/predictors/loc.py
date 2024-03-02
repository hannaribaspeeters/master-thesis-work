import torch
import numpy as np

class LocationPredictor():
    def __init__(self, mode: str = "default"):
        self.mode: str = mode
    
    def sample_location(self, lon: float, lat: float, time: str):
        if self.mode == "default": return torch.tensor([lon, lat], dtype=torch.float32)
        elif self.mode.startswith("cyclical"):
            lon, lat = self._normalize_loc_to_uniform(lon, lat)
            return torch.tensor(self._encode_loc(lon, lat), dtype=torch.float32)
        else: raise ValueError()

    def _normalize_loc_to_uniform(self, lon, lat):
        if self.mode == "cyclical_europe":
            lon = (lon - (-10.53904)) / (34.55792 - (-10.53904))
            lat = (lat - 34.56858) / (71.18392 - 34.56858)
        else:
            raise ValueError
        return lon, lat
    
    def _encode_loc(self, lon, lat):
        features = [np.sin(np.pi * lon), np.cos(np.pi * lon), np.sin(np.pi * lat), np.cos(np.pi * lat)]
        return np.stack(features, axis=-1)
