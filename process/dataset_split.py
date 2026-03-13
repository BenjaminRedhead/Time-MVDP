# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def split_dataset(file_path: str, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Train, val and test ratios must sum to 1.0")

    # Load the dataset
    data = pd.read_csv(file_path)

    # First split off the train set
    train_data, temp_data = train_test_split(data, train_size=train_ratio, random_state=42)

    # Then split the remaining data into val and test
    val_size_adjusted = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(temp_data, train_size=val_size_adjusted, random_state=42)

    return train_data, val_data, test_data

def main(input_dir: str, output_dir: str):
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv") and "train" not in file_name and "val" not in file_name and "test" not in file_name:
            file_path = os.path.join(input_dir, file_name)
            print(f"Processing file: {file_name}")
            
            train_data, val_data, test_data = split_dataset(file_path)

            base_name = os.path.splitext(file_name)[0]
            train_file_name = f"{base_name}_train.csv"
            val_file_name = f"{base_name}_val.csv"
            test_file_name = f"{base_name}_test.csv"

            train_data.to_csv(os.path.join(output_dir, train_file_name), index=False)
            val_data.to_csv(os.path.join(output_dir, val_file_name), index=False)
            test_data.to_csv(os.path.join(output_dir, test_file_name), index=False)

            print(f"Saved {train_file_name}, {val_file_name}, and {test_file_name} to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into train, val, and test sets.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the input files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the split files.')

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
