import numpy as np
import torch

from src.data.predictors.abstract_predictor import AbstractPredictor
from src.utils.location_transforms import scale_to_new_bounds


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
            # Rescaling to [-1,1] before applying sin/cos
            lon, lat = scale_to_new_bounds(
                lon, lat, self.bounds, {"north": 1, "south": -1, "west": -1, "east": 1}
            )
            return torch.tensor(self._encode_loc(lon, lat), dtype=torch.float32)
        else:
            raise ValueError()

    def __str__(self):
        return (
            "Location embedder with mode "
            + self.mode
            + " and bounds "
            + str(self.bounds)
            + " (strict: "
            + str(self.strict)
            + ")"
        )

    def _encode_loc(self, lon, lat):
        features = [
            np.sin(np.pi * lon),
            np.cos(np.pi * lon),
            np.sin(np.pi * lat),
            np.cos(np.pi * lat),
        ]
        return np.stack(features, axis=-1)
