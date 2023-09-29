import pandas as pd
import numpy as np
import tqdm
import pickle
import warnings

from os.path import exists, join
from os import makedirs
from joblib import Parallel, delayed
from datasets.dataloading import load_dataset, get_all_datasets
from single_models import get_single_models, load_models
from tsx.datasets.utils import windowing
from experiments import results_path, lag

warnings.filterwarnings("ignore")

limit_data = None

def load_data(ds_name, ds_index):
    X = load_dataset(ds_name, ds_index)
    X_train_size = int(0.5 * len(X))
    X_val_size = int(0.25 * len(X))

    X_train = X[:X_train_size]
    X_val = X[X_train_size:(X_train_size + X_val_size)]
    X_test = X[(X_train_size+X_val_size):]

    mu = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mu) / std
    X_val = (X_val - mu) / std
    X_test = (X_test - mu) / std

    return X_train, X_val, X_test

def train():

    def _run_parallel(ds_name, ds_index):
        model_names, models = get_single_models()

        makedirs('models', exist_ok=True)
        makedirs(f'models/{ds_name}', exist_ok=True)

        X_train, _, _ = load_data(ds_name, ds_index)
        if np.isnan(X_train).any():
            print(ds_name, ds_index)
            return 
        X, y = windowing(X_train, L=lag)

        for (model_name, model) in zip(model_names, models):

            save_path = f"models/{ds_name}/{ds_index}_{model_name}"
            if exists(save_path):
                continue

            model.fit(X, y)

            with open(save_path, 'wb') as _F:
                pickle.dump(model, _F)

    n_jobs = -1
    Parallel(n_jobs=n_jobs, backend='loky')(delayed(_run_parallel)(ds_name, ds_index) for (ds_name, ds_index) in tqdm.tqdm(get_all_datasets(), desc='train'))

def evaluate():

    def _run_parallel(ds_name, ds_index):

        _, X_val, X_test = load_data(ds_name, ds_index)
        X_val_windowed, _ = windowing(X_val, L=lag)
        X_test_windowed, _ = windowing(X_test, L=lag)

        makedirs('results', exist_ok=True)

        for split in ['val', 'test']:
            if split == 'val':
                X = X_val.copy()
                X_windowed = X_val_windowed.copy()
            else:
                X = X_test.copy()
                X_windowed = X_test_windowed.copy()

            # Load (or create) output csv file
            csv_path = join(results_path, f'{split}_{ds_name}_#{ds_index}.csv')
            if not exists(csv_path):
                prediction_csv = pd.DataFrame({'y': X})
                prediction_csv.to_csv(csv_path)
            else:
                prediction_csv = pd.read_csv(csv_path, header=0, index_col=0)

            models, model_names = load_models(ds_name, ds_index, return_names=True)

            for (model_name, model) in zip(model_names, models):

                preds = model.predict(X_windowed).squeeze()

                preds = np.concatenate([X[:lag].squeeze(), preds])
                prediction_csv[model_name] = preds
                prediction_csv.to_csv(csv_path)

    n_jobs = -1
    Parallel(n_jobs=n_jobs, backend='loky')(delayed(_run_parallel)(ds_name, ds_index) for (ds_name, ds_index) in tqdm.tqdm(get_all_datasets(), desc='eval'))

def train_and_evaluate():
    train()
    evaluate()

def main():
    train_and_evaluate()

if __name__ == "__main__":
    main()
