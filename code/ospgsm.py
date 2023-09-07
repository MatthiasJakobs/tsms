import torch
import numpy as np
import time
import sklearn

from tqdm import tqdm
from tsx.datasets import windowing
from tsx.distances import dtw
from tsx.utils import to_random_state
from shap import TreeExplainer
from sklearn.metrics import mean_squared_error as mse
from seedpy import fixedseed
from captum.attr import KernelShap as CAPTUM_KS
from copy import deepcopy

from train_single_models import load_data
from single_models import load_models
from datasets.dataloading import get_all_datasets

def ks_forward_fn(model, y):
    fn = lambda _x: torch.from_numpy(((model.predict(_x.numpy()).squeeze() - y)**2)).float().sum().reshape(1)
    return fn

class KernelSHAP:

    def __init__(self, model, X_background, y_background, n_samples=25):
        self.model = model
        self.X_background = torch.from_numpy(X_background).float()
        self.y_background = torch.from_numpy(y_background).float()
        self.n_samples = n_samples

    def shap_values(self, X, y):
        all_shaps = []
        for _x, _y in zip(X, y):
            ks = CAPTUM_KS(ks_forward_fn(self.model, _y))
            shaps = ks.attribute(torch.from_numpy(_x.reshape(1, -1)).float(), self.X_background, n_samples=self.n_samples)
            all_shaps.append(shaps.numpy())

        return np.vstack(all_shaps)

def get_explainer(model, X_background, y_background):
    treeshap_compatible = [
        sklearn.tree._classes.DecisionTreeRegressor,
        sklearn.ensemble._gb.GradientBoostingRegressor, 
        sklearn.ensemble._forest.RandomForestRegressor,
    ]
    if type(model) in treeshap_compatible:
        return TreeExplainer(model, data=X_background, feature_perturbation='interventional', model_output='log_loss')
    else:
        return KernelSHAP(model, X_background, y_background)

class OS_PGSM_St_Faster:

    def __init__(self, models, lag, big_lag, threshold=0.5, random_state=None, min_roc_length=7, old_roc=False):
        self.models = models
        self.lag = lag
        self.big_lag = big_lag
        self.threshold = threshold
        self.rng = to_random_state(random_state)
        self.roc_history = []
        self.roc_complete_history = []
        self.min_roc_length = min_roc_length
        self.old_roc = old_roc

    def rebuild(self, X):
        # if self.old_roc:
        #     rocs = [ [] for _ in range(len(self.models))]
        #     x_c, y_c = windowing(X, self.big_lag, z=self.big_lag)
        #     for x, y in zip(x_c, y_c):
        #         cams, best_model = self.evaluate_on_validation(x)
        #         rocs_i, complete_xy = self.calculate_rocs(x, y, cams)
        #         if rocs_i is not None:
        #             rocs[best_model].extend(rocs_i)
        #     return rocs
        # else:

        rocs = [ [] for _ in range(len(self.models))]
        rocs_complete = [ [] for _ in range(len(self.models))]
        _X, _y = windowing(X, self.lag, z=1)

        if not self.old_roc:
            # Find best `n` models
            single_losses = [np.sum((m.predict(_X).squeeze() - _y.squeeze())**2) for m in self.models]
            best_predictors = np.argsort(single_losses)[:3]

            model_backup = deepcopy(self.models)
            self.models = [self.models[idx] for idx in best_predictors]

        x_c, _ = windowing(X, self.big_lag, z=self.big_lag)
        for x in x_c:
            cams, best_model = self.evaluate_on_validation(x)
            _, y = windowing(x, L=15)
            rocs_i, complete_xy = self.calculate_rocs(x, y, cams)
            if rocs_i is not None:
                rocs[best_predictors[best_model]].extend(rocs_i)
                rocs_complete[best_predictors[best_model]].extend(complete_xy)

        if not self.old_roc:
            self.models = model_backup

        return rocs, rocs_complete

    def evaluate_on_validation(self, x_val):
        losses = np.zeros((len(self.models)))
        X, y = windowing(x_val, self.lag, z=1)

        # Find best model according to loss
        losses = [np.sum(((m.predict(X)-y.squeeze())**2).squeeze()) for m in self.models]
        best_forecaster_idx = np.argmin(losses)
        best_forecaster = self.models[best_forecaster_idx]

        # Calculate roc candidates
        explainer = get_explainer(best_forecaster, self.X_background, self.y_background)
        seed = self.rng.randint(0, 1000)
        with fixedseed(np, seed):
            shap_values = explainer.shap_values(X, y)

        return -shap_values, best_forecaster_idx

    def calculate_rocs(self, x, y, cams): 
        def split_array_at_zero(arr):
            indices = np.where(arr != 0)[0]
            splits = []
            i = 0
            while i+1 < len(indices):
                start = i
                stop = start
                j = i+1
                while j < len(indices):
                    if indices[j] - indices[stop] == 1:
                        stop = j
                        j += 1
                    else:
                        break

                if start != stop:
                    splits.append((indices[start], indices[stop]))
                    i = stop
                else:
                    i += 1

            return splits

        rocs = []
        complete_xy = []

        for offset, cam in enumerate(cams):
            max_r = np.max(cam) 
            if max_r == 0:
                continue
            normalized = cam / max_r
            after_threshold = normalized * (normalized > self.threshold)
            if len(np.nonzero(after_threshold)[0]) > 0:
                indidces = split_array_at_zero(after_threshold)
                for (f, t) in indidces:
                    if t-f >= (self.min_roc_length-1):
                        rocs.append(x[f+offset:(t+offset+1)])
                        _x = x[offset:offset+self.lag]
                        _y = y[offset]
                        complete_xy.append(np.concatenate([_x, np.array(_y).reshape(1)]))

        return rocs, complete_xy

    def find_best_forecaster(self, x):
        best_model = -1
        smallest_distance = 1e8

        for i in range(len(self.models)):
            x = x.squeeze()
            for r in self.rocs[i]:
                distance = dtw(r, x)
                if distance < smallest_distance:
                    best_model = i
                    smallest_distance = distance

        return best_model

    # TODO: Make faster
    def forecast_on_test(self, x_test):
        self.test_forecasters = []
        predictions = np.zeros_like(x_test)

        x = x_test[:self.lag]
        predictions[:self.lag] = x

        for x_i in range(self.lag, len(x_test)):
            x = x_test[x_i-self.lag:x_i].reshape(1, -1)

            best_model = self.find_best_forecaster(x)
            self.test_forecasters.append(best_model)
            predictions[x_i] = self.models[best_model].predict(x.reshape(1, -1))

        return np.array(predictions)

    def compute_ranking(self, losses):
        assert len(losses) == len(self.models)
        return np.argmin(losses)

    def run(self, X_train, X_val, X_test):
        X_bg, y_bg = windowing(X_train, L=self.lag)
        self.X_background = X_bg
        self.y_background = y_bg

        self.rocs, self.rocs_complete = self.rebuild(X_val)

        preds = self.forecast_on_test(X_test)
        return preds


class OS_PGSM_Faster(OS_PGSM_St_Faster):
    def __init__(self, models, lag, big_lag, threshold=0.01, min_roc_length=7, old_roc=False):
        super().__init__(models, lag, big_lag, threshold=threshold, min_roc_length=min_roc_length, old_roc=old_roc)

    def detect_concept_drift(self, mu_t, mu_0, R, W, delta=0.99):
        tau = np.sqrt(R**2 * np.log(2/delta) / (2*W))
        return abs(mu_t - mu_0) > tau

    def run(self, X_train, X_val, X_test):
        X_bg, y_bg = windowing(X_train, L=self.lag)
        self.X_background = X_bg
        self.y_background = y_bg

        self.drifts_detected = []
        val_start = 0
        val_stop = len(X_val) + self.lag
        X_complete = np.concatenate([X_val, X_test])
        current_val = X_complete[val_start:val_stop]

        predictions = []

        # Initial creation of ROCs
        self.rocs, self.rocs_complete = self.rebuild(current_val)
        self.test_forecasters = []
        self.roc_history.append(deepcopy(self.rocs))
        self.roc_complete_history.append(deepcopy(self.rocs_complete))

        mu_0 = np.mean(X_val)
        mu_range = [0, self.lag]
        wait_period = 0

        used_range = range(self.lag, len(X_test))

        R = np.max(X_val) - np.min(X_val)

        for target_idx in used_range: 
            f_test = (target_idx-self.lag)
            t_test = (target_idx)
            x = X_test[f_test:t_test] 

            mu_t = np.mean(X_test[mu_range[0]:mu_range[1]])
            W = mu_range[1]-mu_range[0]
            try:
                # Adaptive
                concept_drift = wait_period == 0 and self.detect_concept_drift(mu_t, mu_0, R, W)
            except Exception:
                # Periodic
                concept_drift = int(target_idx % self.periodicity) == 0

            val_start, val_stop = val_start+1, val_stop+1

            if concept_drift:
                self.drifts_detected.append(target_idx)
                val_start = val_stop - len(X_val) - self.lag
                current_val = X_complete[val_start:val_stop]

                mu_0 = mu_t
                wait_period = self.lag

                R = np.max(X_complete[:val_stop]) - np.min(X_complete[:val_stop])
                mu_range = [mu_range[1], mu_range[1]+self.lag]

                new_rocs, new_rocs_complete = self.rebuild(current_val)
                for r_idx, r in enumerate(new_rocs):
                    self.rocs[r_idx].extend(r)
                    self.rocs_complete[r_idx].extend(new_rocs_complete[r_idx])
                self.roc_history.append(deepcopy(self.rocs))
                self.roc_complete_history.append(deepcopy(self.rocs_complete))
            else:
                mu_range[1] += 1
                wait_period = max(0, wait_period-1)

            best_model = self.find_best_forecaster(x)
            
            self.test_forecasters.append(best_model)
            predictions.append(self.models[best_model].predict(x.reshape(1, -1)))

        # if len(self.drifts_detected) != 0:
        #     print('drifts', len(self.drifts_detected))
        return np.concatenate([X_test[:self.lag], np.concatenate(predictions)])


class OS_PGSM_Periodic(OS_PGSM_Faster):

    def __init__(self, models, lag, big_lag, threshold=0.01, min_roc_length=7, old_roc=False, periodicity=None):
        super().__init__(models, lag, big_lag, threshold=threshold, min_roc_length=min_roc_length, old_roc=old_roc)
        self.periodicity = periodicity

    def run(self, X_train, X_val, X_test):
        if self.periodicity is None:
            self.periodicity = int(len(X_test) / 10.0)

        return super().run(X_train, X_val, X_test)

    def detect_concept_drift(self, test_idx):
        return int(test_idx % self.periodicity) == 0

def runtime_measurement():
    data = get_all_datasets()
    # rng = np.random.RandomState(958717)
    # random_indices = rng.choice(np.arange(len(data)), size=15, replace=False)
    # data = [data[idx] for idx in random_indices]

    runtimes = np.zeros((len(data), 3))
    for idx, (ds_name, ds_index) in tqdm(enumerate(data), total=len(data)):
        X_train, X_val, X_test = load_data(ds_name, ds_index)
        models = load_models(ds_name, ds_index, None, False)

        comp = OS_PGSM_St_Faster(models, lag=15, big_lag=25, threshold=0.01, min_roc_length=3)
        before = time.time()
        comp.run(X_train, X_val, X_test)
        after = time.time()
        delta_st = after - before
        runtimes[idx, 0] = delta_st

        comp = OS_PGSM_Faster(models, lag=15, big_lag=25, threshold=0.01, min_roc_length=3)
        before = time.time()
        comp.run(X_train, X_val, X_test)
        after = time.time()
        delta_drift = after - before
        runtimes[idx, 1] = delta_drift

        comp = OS_PGSM_Periodic(models, lag=15, big_lag=25, threshold=0.01, min_roc_length=3)
        before = time.time()
        comp.run(X_train, X_val, X_test)
        after = time.time()
        delta_per = after - before
        runtimes[idx, 2] = delta_per

    mean_runtime = runtimes.mean(axis=0)
    std_runtime = runtimes.std(axis=0)

    for idx, method_name in enumerate(['static', 'driftaware', 'periodic']):
        print(f'{method_name}: {mean_runtime[idx]:.2f} +- {std_runtime[idx]:.2f} [seconds]')

    print(f'estimated over {len(data)} datasets')

def hyperparameter_testing():
    rng = np.random.RandomState(958717)
    data = get_all_datasets()
    random_indices = rng.choice(np.arange(len(data)), size=15, replace=False)
    data = [data[idx] for idx in random_indices]

    # Big Lag, threshold, min_roc_length
    hyperparameter_combos = [
        (25, 0.01, 3),
        (25, 0.05, 3),
        (25, 0.1, 3),
        (25, 0.01, 5),
        (25, 0.05, 5),
        (25, 0.1, 5),
        (25, 0.01, 7),
        (25, 0.05, 7),
        (25, 0.1, 7),
        (40, 0.01, 3),
        (40, 0.05, 3),
        (40, 0.1, 3),
        (40, 0.01, 5),
        (40, 0.05, 5),
        (40, 0.1, 5),
        (40, 0.01, 7),
        (40, 0.05, 7),
        (40, 0.1, 7),
        (50, 0.01, 3),
        (50, 0.05, 3),
        (50, 0.1, 3),
        (50, 0.01, 5),
        (50, 0.05, 5),
        (50, 0.1, 5),
        (50, 0.01, 7),
        (50, 0.05, 7),
        (50, 0.1, 7),
    ]

    loss_matrix = np.zeros((len(data), len(hyperparameter_combos)))

    for row_idx, (ds_name, ds_index) in enumerate(data):
        X_train, X_val, X_test = load_data(ds_name, ds_index)
        models = load_models(ds_name, ds_index, None, False)

        for col_idx, (blag, thresh, min_length) in enumerate(hyperparameter_combos):
            comp = OS_PGSM_Faster(models, lag=15, big_lag=blag, threshold=thresh, old_roc=False)
            comp.min_roc_length = min_length
            preds = comp.run(X_train, X_val, X_test)
            l = mse(X_test, preds, squared=False)
            loss_matrix[row_idx, col_idx] = l

    ranking = np.argsort(np.argsort(loss_matrix, axis=1), axis=1)
    mean_ranks = np.mean(ranking, axis=0)
    print(mean_ranks)
    best_combo = np.argmin(mean_ranks)
    print('best combo', hyperparameter_combos[best_combo])


def main():
    #hyperparameter_testing()
    runtime_measurement()

if __name__ == '__main__':
    main()