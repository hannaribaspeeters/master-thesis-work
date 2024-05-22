import torch
from torch_cluster import grid_cluster
from torch_geometric.nn.pool.consecutive import consecutive_cluster
import pandas as pd
import numpy as np


class GridSampling:
    """ Clusters points into voxels with size :attr:`size`.

    By default, only the last-encountered element of each voxel will be
    kept.

    Parameters
    ----------
    size: float
        Size of a voxel (in each dimension).
    shuffle: bool
        Whether the data should be shuffled before the voxelization.
        Keep to False for deterministic results.
    """

    def __init__(self, size, dim, shuffle=False):
        self.grid_size = size
        # shuffle to avoid that the same point is picked every time I run the script
        self.shuffle = shuffle
        self.dim = dim

    @property
    def _repr_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def __repr__(self):
        attr_repr = ', '.join([f'{k}={v}' for k, v in self._repr_dict.items()])
        return f'{self.__class__.__name__}({attr_repr})'

    def __call__(self, xyz):
        # If 'shuffle' is True, shuffle the points order.
        # Note that voxelization of point attributes will be stochastic
        if self.shuffle:
            idx = torch.randperm(xyz.shape[0], device=xyz.device)
            xyz = xyz[idx]

        # Convert point coordinates to the voxel grid coordinates
        # aixo  es una normalització en les tres dimensions
        coords = torch.round(xyz / self.grid_size)

        # Match each point with a voxel identifier
        # desfa el 3D i dona només un identifier per voxel (i ens és igual l'ordre)
        # de cada voxel es queda només amb l'últim punt
        cluster = grid_cluster(coords, torch.ones(self.dim, device=coords.device))

        # Reindex the clusters to make sure the indices used are
        # consecutive. Basically, we do not want cluster indices to span
        # [0, i_max] without all in-between indices to be used, because
        # this will affect the speed and output size of torch_scatter
        # operations
        cluster, unique_pos_indices = consecutive_cluster(cluster)
        
        cluster_unique_counts = cluster.unique(return_counts=True)[1]
        cluster_ordered_counts = [cluster_unique_counts[i].item() for i in cluster]

        # Use unique_pos_indices to pick values from a single point
        # within each cluster
        #xyz = xyz[unique_pos_indices]

        return unique_pos_indices, cluster_ordered_counts
    
def spherical_to_cartesian(lon, lat, radius_earth):
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    
    x = radius_earth * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius_earth * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius_earth * np.sin(lat_rad)
    
    return x, y, z

def sample_data(data, grid_size, radius_earth=6371):
    # Extract longitude and latitude from the DataFrame
    lon = data['lon'].values
    lat = data['lat'].values
    
    # Convert spherical coordinates to Cartesian coordinates
    x, y, z = spherical_to_cartesian(lon, lat, radius_earth)
    coordinates = torch.stack(
        [torch.tensor(x), torch.tensor(y), torch.tensor(z)],
        dim=1
    )
    
    # Apply grid sampling
    sampler = GridSampling(grid_size, dim=3)
    sampled_data_indices, cluster_density = sampler(coordinates)
    sampled_data = data.iloc[sampled_data_indices].copy()

    return sampled_data, cluster_density

def sample_bioclim_data(dataset, speciesid, grid_size):

    species_data = dataset.data[dataset.data["speciesId"]==speciesid]
    indices = species_data.index
    bioclim_tensors = [dataset[i]["bioclim_pointwise_europe"] for i in indices]
    coordinates = torch.stack(bioclim_tensors, dim=0)

    # Apply grid sampling
    sampler = GridSampling(grid_size, dim=20)
    sampled_data_indices, cluster_density = sampler(coordinates)
    sampled_data = species_data.iloc[sampled_data_indices].copy()
    
    return sampled_data, cluster_density