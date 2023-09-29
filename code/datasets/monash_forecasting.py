
from os.path import exists
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from itertools import product
from os import makedirs

from tsx.datasets.monash import load_monash
from tsx.utils import to_random_state

univariate_datasets = [
    'm4_yearly',
    'm4_quarterly',
    'm4_monthly',
    'm4_weekly',
    'm4_daily',
    'm4_hourly',
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
]


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

def load_dataset(name, idx):
    path = f'code/datasets/series/{name}.npz'
    ds = np.load(path)
    return ds[f'arr_{idx}'].squeeze()

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