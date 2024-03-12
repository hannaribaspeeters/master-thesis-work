import numpy as np
import torch

from src.data.predictors.abstract_predictor import AbstractPredictor
from src.data.predictors.utils.bilinear_interpolate import bilinear_interpolate


class PointwiseBioclimEuropePredictor(AbstractPredictor):
    """European point-wise bioclimate predictor including 19 bioclim variables and elevation."""

    def __init__(self, bioclim_path: str):
        """
        :param bioclim_path: Path pointing to european normalized bioclim raster.
        """
        super().__init__({"north": 72, "south": 34, "west": -11, "east": 35})

        context_feats = np.load(bioclim_path).astype(np.float32)
        self.raster = torch.from_numpy(context_feats)
        self.raster[torch.isnan(self.raster)] = (
            0.0  # replace with mean value (0 is mean post-normalization)
        )

        self.strict = True

    def sample_location(self, lon: float, lat: float, time: str):
        """Sampling the location by first converting to range [-1,1] and then bilinear
        interpolating."""
        if self.strict:
            assert self.check_bounds(lon, lat), (
                "Location " + str((lon, lat)) + "out of bounds for: " + self.__str__()
            )

        # Convert lon and lat both to values between -1 and 1 for bilineare interpolation
        lat = (lat - 34) / (72 - 34)
        lon = (lon - (-11)) / (35 - (-11))
        lon = lon * 2 - 1
        lat = lat * 2 - 1

        vals = bilinear_interpolate(torch.tensor([[lon, lat]]), self.raster)
        return vals.flatten().type(torch.half)

    def __str__(self):
        """Str explaining predictor."""
        return (
            "Bioclimatic embedder for Europe with resolution 1km and bounds "
            + str(self.bounds)
            + " (strict: "
            + str(self.strict)
            + ")"
        )
