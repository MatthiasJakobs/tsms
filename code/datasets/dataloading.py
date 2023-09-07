import numpy as np
import torch
from itertools import product
from datasets.monash_forecasting import load_dataset as load_monash
from datasets.monash_forecasting import generate_monash_data

#limit_data = 15
limit_data = None

monash_names, monash_configs = generate_monash_data()

implemented_datasets = monash_configs

def load_dataset(ds_name, ds_index):
    if ds_name in monash_names:
        return load_monash(ds_name, ds_index)
    raise Exception("Unknown ds name", ds_name)

def get_all_datasets():
    if limit_data is not None:
        rng = np.random.RandomState(958717)
        random_indices = rng.choice(np.arange(len(implemented_datasets)), size=limit_data, replace=False)
        return [implemented_datasets[idx] for idx in random_indices]

    return implemented_datasets
