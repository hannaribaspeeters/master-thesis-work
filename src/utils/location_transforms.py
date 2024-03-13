from typing import Dict

def scale_to_new_bounds(lon:float, lat:float, bounds_before: Dict[str, float], bounds_after: Dict[str, float]) -> tuple[float, float]:
    """
    Scales lon, lat pair to predefined value ranges.
    E.g., the northwestern most points in bounds_before will be returned as the northwestern most point in bounds_after.

    :param lon: lon to be scaled.
    :param lat: lat to be scaled.
    :param bounds_before: Showing the current bounds of the lon/lat given to the function.
    :param bounds_after: signifies the bounds scaled to.
    """
    # First we scale it all to [0,1]
    lon_ret = (lon - bounds_before["west"]) / (bounds_before["east"] - bounds_before["west"])
    lat_ret = (lat - bounds_before["south"]) / (bounds_before["north"] - bounds_before["south"])
    # Then we rescale to the new bounds
    lon_ret = lon_ret * (bounds_after["east"] - bounds_after["west"]) + bounds_after["west"]
    lat_ret = lat_ret * (bounds_after["north"] - bounds_after["south"]) + bounds_after["south"]
    return lon_ret, lat_ret