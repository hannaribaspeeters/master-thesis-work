import argparse
import pandas as pd
from time import time
from thinning_strategies import thin_all_species, thin_majority_species, thin_majority_minority_species

def main(args):
    po_dataset_path = args.input_file
    po_dataset = pd.read_csv(po_dataset_path, sep=";")

    # delete repeated observations
    df = po_dataset.drop_duplicates(
        subset=["speciesId", 'x_EPSG3035', 'y_EPSG3035']
    )
    data = df.copy()

    # Perform spatial thinning based on the specified thinning type
    if args.thinning_strategy == "all":
        thin_all_species(
            data,
            thin_dist=args.thin_dist,
            save=args.save,
            data_dir=args.data_dir)

    elif args.thinning_strategy == "majority":
        thin_majority_species(
            data,
            majority_cutoff=args.majority_cutoff,
            majority_thin_dist=args.majority_thin_dist,
            save=args.save,
            data_dir=args.data_dir
        )
        
    elif args.thinning_strategy == "majority_minority":
        thin_majority_minority_species(
            data,
            majority_cutoff=args.majority_cutoff,
            majority_thin_dist=args.majority_thin_dist,
            minority_thin_dist=args.minority_thin_dist,
            save=args.save,
            data_dir=args.data_dir
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial thinning script")
    parser.add_argument(
        "--thinning_strategy",
        choices=["all", "majority", "majority_minority"],
        default="all",
        help="Type of thinning strategy to perform"
        )

    parser.add_argument(
        "--input_file",
        default="/shares/wegner.ics.uzh/glc23_data/Presences_only_train.csv",
        help="Path to the input CSV file"
    )

    parser.add_argument(
        "--thin_dist",
        type=int,
        default=2, 
        help="Distance for thinning"
    )

    parser.add_argument(
        "--majority_cutoff",
        type=int,
        default=100,
        help="Cutoff for majority species"
    )

    parser.add_argument(
        "--majority_thin_dist",
        type=int,
        default=2,
        help="Distance for thinning majority species"
    )

    parser.add_argument(
        "--minority_thin_dist",
        type=int,
        default=1,
        help="Distance for thinning minority species"
    )

    parser.add_argument(
        "--save",
        type=bool, 
        default=True, 
        help="Save thinned data to file"
    )

    parser.add_argument(
        "--data_dir",
        type=str, 
        default=".", # same directory or /shares/wegner.ics.uzh/glc23_data
        help="Directory where the thinned_data output folder will be stored"
    )

    args = parser.parse_args()
    start_time = time()
    main(args)
    end_time = time()
    print("Execution time:", end_time - start_time, "seconds.")