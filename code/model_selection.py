import numpy as np
from tsx.quantizers import KernelSAX, z_norm

from utils import get_sax_postfix
from single_models import load_models
from roc_tools import drift_detected, split_array_at_zeros
from tsx.datasets.utils import windowing
from tsx.utils import to_random_state
from tsx.distances import dtw
from sklearn.metrics import euclidean_distances, mean_squared_error
from shap import TreeExplainer
from itertools import product
from seedpy import fixedseed
from experiments import lag

class Ensemble:

    def __init__(self, models):
        self.models = models

    def run(self, X_train, X_val, X_test):
        X, y = windowing(X_test, L=lag)
        predictions = np.mean(np.hstack([m.predict(X).reshape(-1, 1) for m in self.models]), axis=-1)
        return np.concatenate([X_test[:lag], predictions.squeeze()])

class BestCaseSelector:

    def __init__(self, models, sax_alphabet_size):
        self.models = models
        self.sax_alphabet_size = sax_alphabet_size

    def run(self, X_train, X_val, X_test):
        X, y = windowing(X_test, L=lag)
        self.test_predictors = []

        preds = []
        for _x, _y in zip(X, y):
            _x = _x.reshape(1, -1)
            model_losses = [(m.predict(_x).squeeze() - _y)**2 for m in self.models]
            best_forecaster = np.argmin(model_losses)
            self.test_predictors.append(best_forecaster)
            preds.append(self.models[best_forecaster].predict(_x).squeeze())

        return np.concatenate([X_test[:lag], np.array(preds)])

class WorstCaseSelector:

    def __init__(self, models, sax_alphabet_size):
        self.models = models

    def run(self, X_train, X_val, X_test):
        X, y = windowing(X_test, L=lag)
        self.test_predictors = []

        preds = []
        for _x, _y in zip(X, y):
            _x = _x.reshape(1, -1)
            model_losses = [(m.predict(_x).squeeze() - _y)**2 for m in self.models]
            worst_forecaster = np.argmax(model_losses)
            self.test_predictors.append(worst_forecaster)
            preds.append(self.models[worst_forecaster].predict(_x).squeeze())

        return np.concatenate([X_test[:lag], np.array(preds)])

class RoCBaseline:

    def __init__(self, models, sax_alphabet_size, random_state=None, dist_fn=euclidean_distances):
        self.models = models
        self.alphabet_size = sax_alphabet_size
        self.rng = to_random_state(random_state)
        self.dist_fn = dist_fn

    def rebuild_rocs(self, X_train, X_val):
        X_val_win, Y_val_win = windowing(X_val, L=lag)

        rocs = [ [] for _ in range(len(self.models))]

        # For now: onlybest
        for x, y in zip(X_val_win, Y_val_win):
            best_model_idx = np.argmin([(m.predict(x.reshape(1, -1)).squeeze() - y)**2 for m in self.models])
            rocs[best_model_idx].append(x.squeeze())

        #print([len(r) for r in rocs])
        return rocs

    def run(self, X_train, X_val, X_test):

        self.test_predictors = []

        # Rebuild
        self.rocs = self.rebuild_rocs(X_train, X_val)

        # Predict
        predictions = []
        X_win, _ = windowing(X_test, L=lag)

        for idx, x in enumerate(X_win):

            # Find best forecaster with closest ROC member to input
            smallest_distance = 10000
            best_model_idx = 0

            for model_idx, roc in enumerate(self.rocs):
                if self.dist_fn == euclidean_distances:
                    tmp = [self.dist_fn(x.reshape(1, -1), r.reshape(1, -1)).squeeze() for r in roc]
                else:
                    tmp = [self.dist_fn(x.reshape(-1), r.reshape(-1)).squeeze() for r in roc]
                if len(tmp) == 0:
                    continue
                min_dist = np.min(tmp)
                if min_dist < smallest_distance:
                    best_model_idx = model_idx
                    smallest_distance = min_dist

            self.test_predictors.append(best_model_idx)

            predictions.append(self.models[best_model_idx].predict(x.reshape(1, -1)).squeeze())

        return np.concatenate([X_test[:lag], np.array(predictions)])

class TShapSelector:

    def __init__(self, models, sax_alphabet_size, random_state=None, dist_fn=euclidean_distances):
        self.models = models
        self.alphabet_size = sax_alphabet_size
        self.rng = to_random_state(random_state)
        self.dist_fn = dist_fn

    def rebuild_rocs(self, X_train, X_val):
        X_val_win, Y_val_win = windowing(X_val, L=lag)
        X_train_win, _ = windowing(X_train, L=lag)

        rocs = [ [] for _ in range(len(self.models))]
        if self.alphabet_size is not None:
            # Generate all possible sax encodings
            mask = np.array(list(product(np.arange(self.alphabet_size), repeat=lag)))
            # Subsample for speed
            size = min(len(mask), 1000)
            mask_indices = self.rng.choice(np.arange(len(mask)), size=size, replace=False)
            X_background = mask[mask_indices]
        else:
            size = min(len(X_train_win), 1000)
            indices = self.rng.choice(np.arange(len(X_train_win)), size=size, replace=False)
            X_background = X_train_win[indices]

        # Create explainers for each model 
        explainers = [TreeExplainer(model, data=X_background, feature_perturbation='interventional', model_output='log_loss') for model in self.models]

        # Needed for RoC creation
        X_orig = X_val_win.copy()

        # Encode if necessary
        if self.alphabet_size is not None:
            sax = KernelSAX(np.arange(self.alphabet_size))
            normalized = z_norm(X_val_win)
            sax.fit(normalized)
            X_val_win = sax.encode(normalized)

        # For now: onlybest
        for idx, (x, y) in enumerate(zip(X_val_win, Y_val_win)):
            best_model_idx = np.argmin([(m.predict(x.reshape(1, -1)).squeeze() - y)**2 for m in self.models])

            seed = self.rng.randint(0, 1000)
            with fixedseed(np, seed):
                shap_values = explainers[best_model_idx].shap_values(x, y).squeeze()

            # Since we explain the loss, we need negative attribution (?)
            saliency = np.maximum(-shap_values, 0)

            if self.dist_fn == dtw:
                #rocs[best_model_idx].append(X_orig[idx].squeeze())
                rocs[best_model_idx].extend(split_array_at_zeros(X_orig[idx].squeeze(), saliency))
                # if len(np.nonzero(saliency)[0]) == len(x):
                #     rocs[best_model_idx].append(X_orig[idx].squeeze())
            else:
                if len(np.nonzero(saliency)[0]) == len(x):
                    rocs[best_model_idx].append(X_orig[idx].squeeze())

        print([len(r) for r in rocs])
        return rocs

    def run(self, X_train, X_val, X_test):

        self.test_predictors = []

        # Rebuild
        self.rocs = self.rebuild_rocs(X_train, X_val)

        # Predict
        predictions = []
        X_win, _ = windowing(X_test, L=lag)

        if self.alphabet_size is not None:
            X_win_enc = z_norm(X_win)
            sax = KernelSAX(np.arange(self.alphabet_size))
            sax.fit(X_win_enc)
            X_win_enc = sax.encode(X_win_enc)

        for idx, x in enumerate(X_win):

            # Find best forecaster with closest ROC member to input
            smallest_distance = 10000
            best_model_idx = 0

            for model_idx, roc in enumerate(self.rocs):
                if self.dist_fn == euclidean_distances:
                    tmp = [self.dist_fn(x.reshape(1, -1), r.reshape(1, -1)).squeeze() for r in roc]
                else:
                    tmp = [self.dist_fn(x.reshape(-1), r.reshape(-1)).squeeze() for r in roc]
                if len(tmp) == 0:
                    continue
                min_dist = np.min(tmp)
                if min_dist < smallest_distance:
                    best_model_idx = model_idx
                    smallest_distance = min_dist

            self.test_predictors.append(best_model_idx)

            if self.alphabet_size is not None:
                inp = X_win_enc[idx].reshape(1, -1)
                pred = self.models[best_model_idx].predict(inp).squeeze()
                predictions.append(pred)
            else:
                predictions.append(self.models[best_model_idx].predict(x.reshape(1, -1)).squeeze())

        return np.concatenate([X_test[:lag], np.array(predictions)])
        
class TShapSelectorDriftaware(TShapSelector):

    def __init__(self, models, sax_alphabet_size, random_state=None, dist_fn=euclidean_distances, enrich_rocs=False):
        self.models = models
        self.alphabet_size = sax_alphabet_size
        self.rng = to_random_state(random_state)
        self.dist_fn = dist_fn
        self.enrich_rocs = enrich_rocs

    def run(self, X_train, X_val, X_test):

        self.test_predictors = []

        # Rebuild
        self.rocs = self.rebuild_rocs(X_train, X_val)

        # Predict
        predictions = []
        X_win, _ = windowing(X_test, L=lag)

        # Setup drift-detection
        val_start = 0
        val_stop = len(X_val) + lag
        X_complete = np.concatenate([X_val, X_test])
        current_val = X_complete[val_start:val_stop]
        means = [ np.mean(current_val) ]
        mean_residuals = []

        if self.alphabet_size is not None:
            X_win_enc = z_norm(X_win)
            sax = KernelSAX(np.arange(self.alphabet_size))
            sax.fit(X_win_enc)
            X_win_enc = sax.encode(X_win_enc)

        for idx, x in enumerate(X_win):

            val_start += 1
            val_stop += 1
            current_val = X_complete[val_start:val_stop]
            means.append(np.mean(current_val))
            mean_residuals.append(means[-1]-means[-2])

            is_drift = drift_detected(mean_residuals, len(current_val), R=1.5)

            if is_drift:
                val_start = val_stop - len(X_val) - lag
                current_val = X_complete[val_start:val_stop]
                mean_residuals = []
                means = [ np.mean(current_val) ]

                if self.enrich_rocs:
                    new_rocs = self.rebuild_rocs(X_train, current_val)
                    for r_idx in range(len(new_rocs)):
                        self.rocs[r_idx].extend(new_rocs[r_idx])
                else:
                    self.rocs = self.rebuild_rocs(X_train, current_val)

            # Find best forecaster with closest ROC member to input
            smallest_distance = 10000
            best_model_idx = 0

            for model_idx, roc in enumerate(self.rocs):
                if self.dist_fn == euclidean_distances:
                    tmp = [self.dist_fn(x.reshape(1, -1), r.reshape(1, -1)).squeeze() for r in roc]
                else:
                    tmp = [self.dist_fn(x.reshape(-1), r.reshape(-1)).squeeze() for r in roc]
                if len(tmp) == 0:
                    continue
                min_dist = np.min(tmp)
                if min_dist < smallest_distance:
                    smallest_distance = min_dist
                    best_model_idx = model_idx

            self.test_predictors.append(best_model_idx)

            if self.alphabet_size is not None:
                inp = X_win_enc[idx].reshape(1, -1)
                pred = self.models[best_model_idx].predict(inp).squeeze()
                predictions.append(pred)
            else:
                predictions.append(self.models[best_model_idx].predict(x.reshape(1, -1)).squeeze())

        return np.concatenate([X_test[:lag], np.array(predictions)])
