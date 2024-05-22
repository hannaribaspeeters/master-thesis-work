import pandas as pd
import numpy as np
from time import time
from src.utils.grid_thinning.grid_sampling_functionalities import (
    sample_data,
    sample_bioclim_data
)
import os

def thin_all_species(
    df,
    thin_dist=1, # in km
    speciesids=None,
    data_dir=".", # current directory
    save=False
):
    timer = time()
    thinned_data_list = []
    cluster_density_list = []

    if speciesids is None:
        speciesids = df["speciesId"].unique()

    counter = 0
    print(f"Start thinning of {len(speciesids)} species...")
    for speciesid in speciesids:
        counter += 1
        start = time()

        species_data = df.loc[df["speciesId"]==speciesid].copy()
        species_thinned_data, cluster_density = sample_data(
            species_data,
            thin_dist
        )

        thinned_data_list.append(species_thinned_data)
        cluster_density_list.append(
            pd.Series(index=species_data.index, data=cluster_density)
        )

        if counter % 200 == 0:
            print(f"Thinned speciesId {counter}/"\
                f"{len(speciesids)} in {np.round(time()-start, 4)} seconds.")

    thinned_data_df = pd.concat(thinned_data_list)
    cluster_density_series = pd.concat(cluster_density_list)
    
    print(f"Completed thinning the data in {np.round(time()-timer, 4)} s.")

    if save: 
        csv_file_path = f"{data_dir}/grid_thinned_data/thin_all/dist_{thin_dist}.csv"
        thinned_data_df.to_csv(csv_file_path, sep=";", index=False)
        print("Thinned data saved successfully to:", csv_file_path)

    return thinned_data_df, cluster_density_series

def thin_majority_species(
    df,
    majority_cutoff=100,
    thin_dist=1, # in km
    save=False,
    data_dir="."
):
    timer = time()
    datasets = []
    cluster_density_list = []

    species_counts = df["speciesId"].value_counts()
    majority_speciesids = species_counts[species_counts >= majority_cutoff].\
        index.tolist()
    
    counter=0
    print(f"Start thinning of {len(majority_speciesids)} majority species...")
    for speciesid in majority_speciesids:
        counter+=1
        species_data = df.loc[df["speciesId"]==speciesid].copy()
        species_thinned_data, cluster_density = sample_data(
            species_data,
            thin_dist
        )
        datasets.append(species_thinned_data)
        cluster_density_list.append(
            pd.Series(index=species_data.index, data=cluster_density)
        )

        if counter % 100 == 0:
            print(f"Thinned species {counter}/{len(majority_speciesids)})")
    minority_species = df[~df['speciesId'].isin(majority_speciesids)].copy()
    datasets.append(minority_species)

    thinned_data = pd.concat(datasets)
    cluster_density_series = pd.concat(cluster_density_list)

    print(f"Completed thinning the data in {np.round(time()-timer, 4)} s.")

    if save: 
        csv_file_path = f"{data_dir}/grid_thinned_data/thin_majority/"\
        f"dist_{thin_dist}_cutoff_{majority_cutoff}.csv"
        thinned_data.to_csv(csv_file_path, sep=";", index=False)
        print("Thinned data saved successfully to:", csv_file_path)

    return thinned_data, cluster_density_series
    
def thin_majority_minority_species(
    df,
    majority_cutoff=100,
    majority_thin_dist=2, #in km
    minority_thin_dist=1,
    save=False,
    data_dir="."
):
    timer = time()
    datasets = []
    species_counts = df["speciesId"].value_counts()
    majority_speciesids = species_counts[species_counts >= majority_cutoff].\
        index.tolist()
    minority_speciesids = species_counts[species_counts < majority_cutoff].\
        index.tolist()
    
    counter = 0
    print(f"Start thinning of {len(majority_speciesids)} majority species...")
    for speciesid in majority_speciesids:
        counter+=1
        species_data = df.loc[df["speciesId"]==speciesid].copy()
        species_thinned_data = sample_data(species_data, majority_thin_dist)
        datasets.append(species_thinned_data)
        if counter % 100 == 0:
            print(f"Thinned species {counter}/{len(majority_speciesids)})")

    counter = 0
    print(f"Start thinning of {len(minority_speciesids)} minority species...")
    for speciesid in minority_speciesids:
        counter+=1
        species_data = df.loc[df["speciesId"]==speciesid].copy()
        species_thinned_data = sample_data(species_data, minority_thin_dist)
        datasets.append(species_thinned_data)
        if counter % 100 == 0:
            print(f"Thinned species {counter}/{len(minority_speciesids)})")

    thinned_data = pd.concat(datasets)

    print(f"Completed thinning the dataframe in {np.round(time()-timer, 4)} seconds.")

    if save:
        csv_file_path = f"{data_dir}/grid_thinned_data/thin_majority_minority/"\
        f"majdist_{majority_thin_dist}_mindist_{minority_thin_dist}_cutoff"\
        f"_{majority_cutoff}.csv"
        thinned_data.to_csv(csv_file_path, sep=";", index=False)
        print("Thinned data saved successfully to:", csv_file_path)

    return thinned_data

def thin_bioclim_all_species(
    dataset,
    thin_dist=1, # in km
    speciesids=None,
    save=False,
):
    timer = time()
    thinned_data_list = []
    cluster_density_list = []

    speciesids = dataset.data["speciesId"].unique()
    counter = 0
    
    print(f"Start thinning of {len(speciesids)} species...")
    for speciesid in speciesids:
        counter += 1
        start = time()

        species_data = dataset.data.loc[dataset.data["speciesId"]==speciesid].\
            copy()

        species_thinned_data, cluster_density = sample_bioclim_data(
            dataset,
            speciesid,
            thin_dist
        )

        thinned_data_list.append(species_thinned_data)

        cluster_density_list.append(
            pd.Series(index=species_data.index, data=cluster_density)
        )

        if counter % 200 == 0:
            print(f"Thinned speciesId {counter}/"\
                f"{len(speciesids)} in {np.round(time()-start, 4)} seconds.")
    
    
    thinned_data_df = pd.concat(thinned_data_list)
    cluster_density_series = pd.concat(cluster_density_list)
    
    print(f"Completed thinning the data in {np.round(time()-timer, 4)} s.")
    print(f"Reduced the dataset from size {len(dataset.data)} to {len(thinned_data_df)}.")

    if save: 
        csv_file_path = f"data/bioclim_thinned/"\
        f"dist_{thin_dist}.csv"
        thinned_data_df.to_csv(csv_file_path, sep=";", index=False)
        print("Thinned data saved successfully to:", csv_file_path)

    return thinned_data_df, cluster_density_series
