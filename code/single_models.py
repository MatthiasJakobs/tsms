import pickle
import numpy as np

from seedpy import fixedseed
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted
from itertools import product
from utils import get_sax_postfix

def load_models(ds_name, ds_index, sax_alphabet_size=None, return_names=False):
    loaded_models = []
    model_names, models = get_single_models()

    postfix = get_sax_postfix(sax_alphabet_size)

    for (model_name, model) in zip(model_names, models):
        save_path = f'models/{ds_name}/{ds_index}_{model_name}_{postfix}'
        with open(save_path, 'rb') as _F:
            model = pickle.load(_F)

        check_is_fitted(model)
        loaded_models.append(model)

    if return_names:
        return loaded_models, model_names
    return loaded_models

def get_single_models():
    with fixedseed(np, seed=20230306):
        models = []
        model_names = []

        # Single decision trees
        _max_depth = [4, 8, 16]
        _min_samples_leaf = [1]
        for idx, (max_depth, min_samples_leaf) in enumerate(product(_max_depth, _min_samples_leaf)):
            models.append(DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf))
            model_names.append(f'dt-{idx}')

        # Random Forests 
        _max_depth = [2, 4, 6]
        _n_estimators = [16, 32, 64]
        for idx, (max_depth, n_estimators) in enumerate(product(_max_depth, _n_estimators)):
            models.append(RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators))
            model_names.append(f'rf-{idx}')

        # GBR
        _max_depth = [2, 4, 6]
        _n_estimators = [16, 32, 64]
        for idx, (max_depth, n_estimators) in enumerate(product(_max_depth, _n_estimators)):
            models.append(GradientBoostingRegressor(max_depth=max_depth, n_estimators=n_estimators))
            model_names.append(f'gbr-{idx}')

    return model_names, models

if __name__ == '__main__':
    names, models = get_single_models()
    for idx, name in enumerate(names):
        print(idx, name)
    print(len(names))
