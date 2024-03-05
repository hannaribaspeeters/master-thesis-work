import numpy as np
import torch

from src.data.predictors.abstract_predictor import AbstractPredictor


class LocationPredictor(AbstractPredictor):
    def __init__(self, mode: str = "default"):
        if mode == "default":
            super().__init__({"north": 90, "south": -90, "west": -180, "east": 180})
            # The self.strict value defines whether to make the bounds check or not
            self.strict = False
        if mode == "cyclical_europe":
            super().__init__(
                {"north": 71.18392, "south": 34.56858, "west": -10.53904, "east": 34.55792}
            )
            self.strict = True

        self.mode: str = mode

    def sample_location(self, lon: float, lat: float, time: str):
        if self.strict:
            assert self.check_bounds(lon, lat)

        if self.mode == "default":
            return torch.tensor([lon, lat], dtype=torch.float32)
        elif self.mode.startswith("cyclical"):
            lon, lat = self._normalize_loc_to_uniform(lon, lat)
            return torch.tensor(self._encode_loc(lon, lat), dtype=torch.float32)
        else:
            raise ValueError()

    def __str__(self):
        return "Location embedder with mode " + self.mode + " and bounds " + str(self.bounds) + " (strict: " + str(self.strict) + ")"

    def _normalize_loc_to_uniform(self, lon, lat):
        if self.mode == "cyclical_europe":
            lon = (lon - (-10.53904)) / (34.55792 - (-10.53904))
            lat = (lat - 34.56858) / (71.18392 - 34.56858)
        else:
            raise ValueError
        return lon, lat

    def _encode_loc(self, lon, lat):
        features = [
            np.sin(np.pi * lon),
            np.cos(np.pi * lon),
            np.sin(np.pi * lat),
            np.cos(np.pi * lat),
        ]
        return np.stack(features, axis=-1)
