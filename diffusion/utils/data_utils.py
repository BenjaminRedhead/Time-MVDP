# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pandas as pd
import numpy as np
from einops import rearrange
from distutils.util import strtobool
from datetime import datetime
from pathlib import Path
from diffusion.ldm.data.tsg_dataset import load_data_from_file
import os
from pathlib import Path

prefix = ''
if 'DATA_ROOT' in os.environ and os.path.exists(os.environ['DATA_ROOT']):
    prefix = Path(os.environ['DATA_ROOT'])
else:
    print("DATA_ROOT not exist or not defined!")

test_data_map = {
    # 'solar': 'solar_{seq_len}_val.npy',
    # 'electricity': 'electricity_{seq_len}_val.npy',
    # 'traffic': 'traffic_{seq_len}_val.npy',
    # 'kddcup': 'kddcup_{seq_len}_val.npy',
    # 'taxi': 'taxi_{seq_len}_val.npy',
    # 'exchange': 'exchange_{seq_len}_val.npy',
    # 'fred_md': 'fred_md_{seq_len}_val.npy',
    # 'nn5': 'nn5_{seq_len}_val.npy',
    # 'web': 'web_{seq_len}_test_sample.npy',
    # 'stock': 'stock_{seq_len}_test_sample.npy',
    # 'temp': 'temp_{seq_len}_val.npy',
    # 'rain': 'rain_{seq_len}_val.npy',
    # 'pedestrian': 'pedestrian_{seq_len}_val.npy',
    # 'wind_4_seconds': 'wind_4_seconds_{seq_len}_val.npy'
}

test_data_map_text = {
    #'solar': '{DATA_ROOT}/solar_numeric_text.npy'
    #'electricity': '{DATA_ROOT}/electricity_numeric_text.npy',
    #'traffic': '{DATA_ROOT}/traffic_numeric_text.npy',
    #'kddcup': '{DATA_ROOT}/kddcup_numeric_text.npy',
    #'taxi': '{DATA_ROOT}/taxi_numeric_text.npy',
    #'exchange': '{DATA_ROOT}/exchange_numeric_text.npy',
    #'fred_md': '{DATA_ROOT}/fred_md_numeric_text.npy,
    'nn5': '{DATA_ROOT}/nn5_with_descriptions_168.csv'
    #'pedestrian': '{DATA_ROOT}/pedestrian_numeric_text.npy',
    #'wind_4_seconds': "{DATA_ROOT}/wind_4_seconds_numeric_text.npy",
    #'rain': '{DATA_ROOT}/rain_numeric_text.npy',
    #'temp': '{DATA_ROOT}/temp_numeric_text.npy'
}

def test_data_loading(data_name, seq_len, stride=1, univar=False, use_text=True, normalize='centered_pit'):

    """
    Loads test data for evaluation or zero-shot generation.

    Args:
        data_name (str): Dataset name (used to look up test_data_map_text / test_data_map).
        seq_len (int): Sequence length (window).
        stride (int): Step size (only for multivariate cases).
        univar (bool): Whether to force univariate reshaping.
        use_text (bool): Whether to load text descriptions and process embeddings.

    Returns:
        numeric_data (np.ndarray): Shape (n, seq_len, 1)
        text_data (np.ndarray): Shape (n, embedding_dim) if use_text=True else None
    """

    global prefix

    if use_text and data_name in test_data_map_text:
        data_path = test_data_map_text[data_name]

        assert prefix is not None, "DATA_ROOT (prefix) must be defined!"
        data_path = data_path.replace('{DATA_ROOT}', str(prefix))

        raw_numeric_data, _ = load_data_from_file(data_path, use_text=False)

        raw_numeric_data = np.array(raw_numeric_data)
        if raw_numeric_data.ndim == 2:
            raw_numeric_data = raw_numeric_data.reshape(-1, raw_numeric_data.shape[1], 1)

        print(f"[TEST DEBUG] Raw numeric data loaded: shape = {raw_numeric_data.shape}")

        normalizer = fit_normalizer_static(raw_numeric_data, normalize=normalize)

        numeric_data, text_embeddings = load_data_from_file(
            data_path,
            use_text=True,
            normalizer=normalizer,
            normalize_fn=lambda data, norm: transform_static(data, normalizer=norm, normalize=normalize)
        )


        numeric_data = np.array(numeric_data)
        if numeric_data.ndim == 2:
            numeric_data = numeric_data.reshape(-1, numeric_data.shape[1], 1)

        print(f"[TEST DATA] Loaded CSV:")
        print(f"  numeric_data.shape = {numeric_data.shape}")
        print(f"  text_embeddings.shape = {text_embeddings.shape}")

        return numeric_data, text_embeddings

    elif data_name in test_data_map:
        data_path = prefix / test_data_map[data_name].format(seq_len=seq_len, stride=stride)

        ori_data = np.load(data_path) 

        if univar:
            ori_data = rearrange(ori_data, 'n t c -> (n c) t 1')
            print(f"[TEST DATA] Univar mode enabled.")

        print(f"[TEST DATA] Loaded non-text dataset:")
        print(f"  ori_data.shape = {ori_data.shape}")

        return ori_data, None

    else:
        raise ValueError(f"Unknown dataset name: {data_name}")



def fit_normalizer_static(data: np.ndarray, normalize='centered_pit'):
    data = data.flatten()
    normalizer = {}

    if normalize == 'zscore':
        normalizer['mean'] = np.nanmean(data)
        normalizer['std'] = np.nanstd(data)
    elif normalize == 'robust_iqr':
        normalizer['median'] = np.median(data)
        normalizer['iqr'] = np.subtract(*np.percentile(data, [75, 25]))
    elif normalize == 'robust_mad':
        normalizer['median'] = np.median(data)
        normalizer['mad'] = np.median(np.abs(data - normalizer['median']))
    elif normalize == 'minmax':
        normalizer['min'] = np.nanmin(data)
        normalizer['max'] = np.nanmax(data)
    elif normalize in ['pit', 'centered_pit']:
        normalizer['ecdf'] = ECDF(data)
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")

    return normalizer

def transform_static(data: np.ndarray, normalizer=None, normalize='centered_pit'):
    assert normalizer is not None, "Must provide normalizer."

    if normalize == 'zscore':
        return (data - normalizer['mean']) / (normalizer['std'] + 1e-8)
    elif normalize == 'robust_iqr':
        return (data - normalizer['median']) / (normalizer['iqr'] + 1e-8)
    elif normalize == 'robust_mad':
        return (data - normalizer['median']) / (normalizer['mad'] + 1e-8)
    elif normalize == 'minmax':
        return (data - normalizer['min']) / (normalizer['max'] - normalizer['min'] + 1e-8)
    elif normalize in ['pit', 'centered_pit']:
        data_shape = data.shape
        norm_data = normalizer['ecdf'](data.flatten()).reshape(data_shape)
        if normalize == 'centered_pit':
            norm_data = norm_data * 2 - 1
        return norm_data
    else:
        raise ValueError(f"Unknown normalization method: {normalize}")


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )
