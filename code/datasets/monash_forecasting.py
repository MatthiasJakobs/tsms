# code from https://github.com/rakshitha123/TSForecasting/blob/master/utils/data_loader.py

from datetime import datetime
#from numpy import distutils
from os.path import exists
import matplotlib.pyplot as plt
import hashlib
import distutils
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from itertools import product
from os import makedirs

from tsx.datasets.monash import load_monash
from tsx.utils import to_random_state

ts_path = "code/datasets/monash_complete.npy"

univariate_datasets = [
    'm4_yearly',
    'm4_quarterly',
    'm4_monthly',
    'm4_weekly',
    'm4_daily',
    'm4_hourly',
    #'tourism_yearly',
    'tourism_quarterly',
    'tourism_monthly',
    'cif_2016',
    'australian_electricity_demand',
    'dominick',
    'bitcoin_nomissing',
    'pedestrian_counts',
    'vehicle_trips_nomissing',
    'kdd_cup_nomissing',
    'weather',
    # 'sunspot_nomissing',
    # 'saugene_river_flow',
    # 'us_births',
]


# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
def convert_tsf_to_dataframe(full_file_path_and_name, replace_missing_vals_with = 'NaN', value_column_name = "series_value"):
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

    with open(full_file_path_and_name, 'r', encoding='cp1252') as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"): # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (len(line_content) != 3):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if len(line_content) != 2:  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(distutils.util.strtobool(line_content[1]))
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(distutils.util.strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception("Missing attribute section. Attribute section must come before data.")

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception("Missing attribute section. Attribute section must come before data.")
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

                        if(len(series) == 0):
                            raise Exception("A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol")

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if (numeric_series.count(replace_missing_vals_with) == len(numeric_series)):
                            raise Exception("All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series.")

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(full_info[i], '%Y-%m-%d %H-%M-%S')
                            else:
                                raise Exception("Invalid attribute type.") # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if(att_val == None):
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

        return loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length

def extract_subseries(df, amount=15, length=500, min_length=250, random_state=0):
    extracted_timeseries = []
    n_rows = len(df)

    rng = to_random_state(random_state)
    counter = 0
    for row in range(n_rows):
        subseries = df.iloc[row]['series_value'].to_numpy().squeeze()

        if len(subseries) <= min_length:
            continue

        if np.any(np.isnan(subseries)):
            continue

        subseries_length = min(length, len(subseries))
        padding = len(subseries) - subseries_length
        if len(subseries) == subseries_length or padding <= 1:
            start_idx = 0
        else:
            start_idx = rng.randint(0, high=padding-1)

        series = subseries[start_idx:(start_idx+subseries_length)]

        if np.mean(series) == series[0]:
            continue

        scaler = StandardScaler()
        series = scaler.fit_transform(series.reshape(-1, 1))

        if np.sum(series) == 0:
            continue

        halfway_point = int(len(series) * 0.5)

        if np.all(series[:halfway_point] == series[0]):
            continue

        extracted_timeseries.append(series.squeeze())
        counter += 1

    indices = rng.choice(len(extracted_timeseries), size=min(amount, len(extracted_timeseries)))
    return [extracted_timeseries[i] for i in indices]

def calculate_dataset_seed(ds_name):
    return int(hashlib.md5((ds_name).encode("utf-8")).hexdigest(), 16) & 0xffffffff

def describe_dataset(used_dataset_names, dataset):
    print("Used datasets")
    for name in used_dataset_names:
        print(name)
    print(f"Dataset size {dataset.shape}")
    
def load_dataset(name, idx):
    path = f'code/datasets/series/{name}.npz'
    ds = np.load(path)
    return ds[f'arr_{idx}'].squeeze()

def plot_datasets():
    full_ds = np.load(ts_path) 
    for idx in range(25):
        ds = full_ds[idx].squeeze()

        plt.figure(figsize=(10, 4))
        plt.plot(ds)
        plt.savefig(f"plots/ds_{idx}.png")
        plt.close()

def generate_monash_data():

    rng = np.random.RandomState(23817)

    n_timeseries = 1600
    n_per_dataset = int(np.ceil(n_timeseries / len(univariate_datasets)))

    makedirs('code/datasets/series', exist_ok=True)

    configs = []
    names = []
    summary = []
    for ds_name in univariate_datasets:
        dataset_path = f'code/datasets/series/{ds_name}.npz'
        if not exists(dataset_path):
            X = load_monash(ds_name)
            ts = extract_subseries(X, amount=n_per_dataset, random_state=rng)
            if len(ts) == 0:
                continue

            assert np.all([np.isnan(x).any() == False for x in ts])

            np.savez(dataset_path, *ts, length=len(ts))
            print('Preprocessed', ds_name)

        ds = np.load(dataset_path)
        configs.extend(list(product([ds_name], np.arange(ds['length']))))
        names.append(ds_name)
        summary.append([ds_name, n_per_dataset])

    return names, configs


if __name__ == "__main__":
    names, _ = generate_monash_data()
    amount = 0
    for ds_name in names:
        ds = np.load(f'code/datasets/series/{ds_name}.npz')
        lengths = [len(ds[f'arr_{idx}']) for idx in range(len(ds)-1)]
        min_length = np.min(lengths)
        max_length = np.max(lengths)
        print(ds_name, len(ds)-1, (min_length, max_length))
        amount += len(ds)-1

    print(amount)
    exit()

    used_datasets = sorted(glob.glob("code/datasets/monash_ts/*.tsf"))

    if exists(ts_path):
        dataset = np.load(ts_path)
        describe_dataset(used_datasets, dataset)
        exit()

    ts_agg = []
    for path in used_datasets:
        dataset_name = path.split("/")[-1].split(".")[0]
        dataset_seed = calculate_dataset_seed(dataset_name)
        print(dataset_name, dataset_seed)
        loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(path)
        ts = extract_subseries(loaded_data, random_state=dataset_seed)
        ts_agg.append(ts)

    dataset = np.concatenate(ts_agg)
    np.save(ts_path, dataset)
    describe_dataset(used_datasets, dataset)

