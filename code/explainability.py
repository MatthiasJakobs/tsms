import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from tsx.datasets import windowing
from tsx.distances import dtw
from shap import TreeExplainer
from fastdtw import fastdtw
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
from os import makedirs
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from datasets.dataloading import get_all_datasets
from train_single_models import load_data, load_models
from ospgsm import OS_PGSM_St_Faster, OS_PGSM_Faster

def set_style(subplots, autolayout=True, height_scale=1):
    # Using seaborn's style
    plt.style.use('seaborn')

    tex_fonts = {
        "lines.linewidth": 1,
        # Use LaTeX to write all text
        #"axes.linewidth": 8,
        "text.usetex": True,
        "figure.autolayout": autolayout,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        # "legend.fontsize": 8,
        # "xtick.labelsize": 8,
        # "ytick.labelsize": 8
        "legend.fontsize": 6,
        "legend.frameon": "True",
        "xtick.labelsize": 6,
        "ytick.labelsize": 6
    }

    plt.rcParams.update(tex_fonts)

    # 252.0pt
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27
    fig_width_in = inches_per_pt * 252.0
    golden_ratio = (5**.5 - 1) / 2

    fig_height_in = height_scale * fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return fig_width_in, fig_height_in

def unique_rows(matrix):
    already_seen = []
    to_return = []
    for x in matrix:
        if len(already_seen) == 0:
            already_seen.append(x)
            to_return.append(x)
        else:
            _as = np.vstack(already_seen)
            if not np.any(np.all(x == _as, axis=1)):
                already_seen.append(x)
                to_return.append(x)

    return np.vstack(to_return)

def drift_model_no_change(ds_name, ds_index):
    print(ds_name, ds_index)
    X_train, X_val, X_test = load_data(ds_name, ds_index)
    models = load_models(ds_name, ds_index, sax_alphabet_size=None, return_names=False)

    comp = OS_PGSM_Faster(models, lag=15, big_lag=25, threshold=0.01, min_roc_length=3)
    comp.run(X_train, X_val, X_test)

    print('drifts', comp.drifts_detected)
    for drift_idx, d_idx in enumerate(comp.drifts_detected):
        #d_idx = comp.drifts_detected[drift_idx]
    
        # -----------------------
        X_test_win, y_test_win = windowing(X_test, L=15)
        x_test, y_test = X_test_win[d_idx], y_test_win[d_idx]

        # Before
        idx_before = np.argmin([min([dtw(r, x_test) for r in roc]) if len(roc) > 0 else 100000 for roc in comp.roc_history[drift_idx]])
        pred_before = models[idx_before].predict(x_test.reshape(1, -1)).squeeze()
        idx_after = np.argmin([min([dtw(r, x_test) for r in roc]) if len(roc) > 0 else 100000 for roc in comp.roc_history[drift_idx+1]])
        pred_after = models[idx_after].predict(x_test.reshape(1, -1)).squeeze()

        if idx_before == idx_after:
            return

        subplots = (3, 1)
        fig_width_in, fig_height_in = set_style(subplots)

        f, (ax1, ax2, ax3) = plt.subplots(subplots[0], subplots[1], figsize=(fig_width_in, fig_height_in), sharey=True)

        ax1.plot(x_test, color='black', label='Window to forecast')
        #ax1.plot([len(x_test)-1, len(x_test)], [x_test[-1], y_test], linestyle='--', color='black', label='Ground truth')
        ax1.set_title('Datapoint to forecast')

        rng = np.random.RandomState(8278167)

        # --------------------------------
        rocs = deepcopy(comp.roc_history[drift_idx][idx_before])
        rocs_complete = np.array(deepcopy(comp.roc_complete_history[drift_idx][idx_before]))

        x_pos_label = np.array([[14, 15]]).repeat(len(rocs_complete), 0)
        x_roc_complete = rocs_complete[:, :-1]

        closest_roc = np.argmin([dtw(r, x_test) for r in rocs])
        ax2.set_title('$R_{' + str(idx_after) + '}$ before drift')

        subset = rng.choice(len(x_roc_complete), size=min(5, len(x_roc_complete)), replace=False)
        
        buffer = []
        for x in x_roc_complete[subset]:
            if len(buffer) == 0:
                ax2.plot(x, color='black', alpha=0.2)
            else:
                if not np.any(np.all(x == np.vstack(buffer), axis=1)):
                    ax2.plot(x, color='black', alpha=0.2)
            buffer.append(x)
        already_plotted = deepcopy(x_roc_complete[subset])

        labeled = False
        for r_idx, r in enumerate(rocs):
            corr_x = x_roc_complete[r_idx]
            # Find matching
            len_r = len(r)
            
            index = 0
            min_dist = 10000000
            for i in range(len(corr_x-len_r)):
                dist = dtw(corr_x[i:i+len_r], r)
                if dist < min_dist:
                    index = i
                    min_dist = dist

            if r_idx == closest_roc:
                ax2.plot(corr_x, color='black', alpha=0.5)
                ax2.plot(list(range(index, index+len_r)), r, alpha=1, color='red', label='Closest before drift')
                already_plotted = np.vstack([already_plotted, x_roc_complete[r_idx]])
            else:
                if r_idx in subset:
                    if labeled:
                        ax2.plot(list(range(index, index+len_r)), r, alpha=0.3, color='blue')
                    else:
                        ax2.plot(list(range(index, index+len_r)), r, alpha=0.3, color='blue', label='RoC member')
                        labeled = True


        # --------------------------------
        buffer = []
        for x in x_roc_complete[subset]:
            if len(buffer) == 0:
                ax3.plot(x, color='black', alpha=0.2)
            else:
                if not np.any(np.all(x == np.vstack(buffer), axis=1)):
                    ax3.plot(x, color='black', alpha=0.2)
            buffer.append(x)
        already_plotted = np.vstack([already_plotted, x_roc_complete[subset]])

        rocs = deepcopy(comp.roc_history[drift_idx+1][idx_after])
        rocs_complete = np.array(deepcopy(comp.roc_complete_history[drift_idx+1][idx_after]))

        x_pos_label = np.array([[14, 15]]).repeat(len(rocs_complete), 0)
        x_roc_complete = rocs_complete[:, :-1]

        closest_roc = np.argmin([dtw(r, x_test) for r in rocs])

        ax3.set_title('$R_{' + str(idx_after) + '}$ after drift')
        subset = rng.choice(len(x_roc_complete), size=5, replace=False)

        for i in subset:
            x = x_roc_complete[i]
            if not np.any(np.all((x == already_plotted), axis=1)):
                ax3.plot(x, color='black', alpha=0.2)

        # Only plot new rocs:
        labeled = False
        for r_idx, r in enumerate(rocs):
            corr_x = x_roc_complete[r_idx]
            # Find matching
            len_r = len(r)
            
            index = 0
            min_dist = 10000000
            for i in range(len(corr_x-len_r)):
                dist = dtw(corr_x[i:i+len_r], r)
                if dist < min_dist:
                    index = i
                    min_dist = dist

            if r_idx == closest_roc:
                if not np.any(np.all((x_roc_complete[r_idx] == already_plotted), axis=1)):
                    ax3.plot(x_roc_complete[r_idx], color='black', alpha=0.5)
                ax3.plot(list(range(index, index+len_r)), r, alpha=1, color='orange', label='Closest after drift')
            else:
                if r_idx in subset:
                    ax3.plot(list(range(index, index+len_r)), r, alpha=0.3, color='blue')


        #f.legend(bbox_to_anchor=(1.19, 0.70), borderaxespad=0.0)
        
        x_min = 7
        x_max = 15
        ax1.set_xticks(np.arange(0, 15, 2), np.arange(1, 16, 2)[::-1])
        ax2.set_xticks(np.arange(0, 15, 2), np.arange(1, 16, 2)[::-1])
        ax3.set_xticks(np.arange(0, 15, 2), np.arange(1, 16, 2)[::-1])
        ax1.set_xlabel('lag')
        ax2.set_xlabel('lag')
        ax3.set_xlabel('lag')
        f.suptitle('Change in closest subseries for model $f_{' + str(idx_after) + '}$')
        f.savefig(f'plots/drift_no_change_{ds_name}_{ds_index}_{drift_idx}.png')
        f.savefig(f'plots/drift_no_change_{ds_name}_{ds_index}_{drift_idx}.pdf')

def drift_model_change():

    candidates = [
        ['m4_yearly', 1, 2, 14],
    ]

    for (ds_name, ds_index, drift_idx, data_idx) in candidates:

        X_train, X_val, X_test = load_data(ds_name, ds_index)
        models = load_models(ds_name, ds_index, sax_alphabet_size=None, return_names=False)

        comp = OS_PGSM_Faster(models, lag=15, big_lag=25, threshold=0.01, min_roc_length=3)
        comp.run(X_train, X_val, X_test)
        X_test_win, y_test_win = windowing(X_test, L=15)

        drifts_detected = comp.drifts_detected
        if len(drifts_detected) == 0:
            continue
        x_test, y_test = X_test_win[data_idx], y_test_win[data_idx]

        idx_before = np.argmin([min([dtw(r, x_test) for r in roc]) if len(roc) > 0 else 100000 for roc in comp.roc_history[drift_idx]])
        idx_after = np.argmin([min([dtw(r, x_test) for r in roc]) if len(roc) > 0 else 100000 for roc in comp.roc_history[drift_idx+1]])

        pred_before = models[idx_before].predict(x_test.reshape(1, -1)).squeeze()
        pred_after = models[idx_after].predict(x_test.reshape(1, -1)).squeeze()

        subplots = (3, 1)
        fig_width_in, fig_height_in = set_style(subplots)

        f, (ax1, ax2, ax3) = plt.subplots(subplots[0], subplots[1], figsize=(fig_width_in, fig_height_in))

        ax1.plot(x_test, color='black', label='Window to forecast')
        ax1.plot([len(x_test)-1, len(x_test)], [x_test[-1], y_test],linestyle='--',  color='green', label='Ground truth')
        ax1.plot([len(x_test)-1, len(x_test)], [x_test[-1], pred_before], linestyle='--', color='red', label='Prediction $f_{' + str(idx_before) + '}$')
        ax1.plot([len(x_test)-1, len(x_test)], [x_test[-1], pred_after], linestyle='--', color='orange', label='Prediction $f_{' + str(idx_after) + '}$')
        ax1.set_title('Prediction using selected models')
        ax1.set_xlabel('$t$')
        ax1.set_xticks([])


        rng = np.random.RandomState(192857)

        # --------------------------------
        rocs = comp.roc_history[drift_idx][idx_before]
        rocs_complete = np.array(comp.roc_complete_history[drift_idx][idx_before])

        x_pos_label = np.array([[14, 15]]).repeat(len(rocs_complete), 0)
        x_roc_complete = rocs_complete[:, :-1]

        closest_roc = np.argmin([dtw(r, x_test) for r in rocs])
        ax2.set_title('RoCs of $f_{' + str(idx_before) + '}$ (best before drift)')

        subset = rng.choice(len(x_roc_complete), size=5, replace=False)

        remaining = len(unique_rows(x_roc_complete[subset]))
        ax2.plot(unique_rows(x_roc_complete[subset]).T, color='black', alpha=0.2)
        ax2.plot(x_pos_label[:remaining].T, unique_rows(rocs_complete[subset, -2:]).T, alpha=0.2, color='black', linestyle='--')
        ax2.set_xlabel('$t$')

        labeled = False
        for r_idx, r in enumerate(rocs):
            corr_x = x_roc_complete[r_idx]
            # Find matching
            len_r = len(r)
            
            index = 0
            min_dist = 10000000
            for i in range(len(corr_x-len_r)):
                dist = dtw(corr_x[i:i+len_r], r)
                if dist < min_dist:
                    index = i
                    min_dist = dist

            if r_idx == closest_roc:
                ax2.plot(x_roc_complete[r_idx], color='black', alpha=0.5)
                ax2.plot(x_pos_label[0], rocs_complete[r_idx, -2:].T, alpha=0.5, color='black', linestyle='--')
                ax2.plot(list(range(index, index+len_r)), r, alpha=1, color='red', label='Closest in $R_{' + str(idx_before) + '}$')
            else:
                if r_idx in subset:
                    if labeled:
                        ax2.plot(list(range(index, index+len_r)), r, alpha=0.3, color='blue')
                    else:
                        ax2.plot(list(range(index, index+len_r)), r, alpha=0.3, color='blue', label='RoC member')
                        labeled = True

        ax2.set_xticks([])

        # --------------------------------
        rocs = comp.roc_history[drift_idx+1][idx_after]
        rocs_complete = np.array(comp.roc_complete_history[drift_idx+1][idx_after])

        x_pos_label = np.array([[14, 15]]).repeat(len(rocs_complete), 0)
        x_roc_complete = rocs_complete[:, :-1]

        closest_roc = np.argmin([dtw(r, x_test) for r in rocs])

        ax3.set_title('RoCs of $f_{' + str(idx_after) + '}$ (best after drift)')
        ax3.set_xlabel('$t$')
        subset = rng.choice(len(x_roc_complete), size=5, replace=False)
        remaining = len(unique_rows(x_roc_complete[subset]))
        ax3.plot(unique_rows(x_roc_complete[subset]).T, color='black', alpha=0.2)
        ax3.plot(x_pos_label[:remaining].T, unique_rows(rocs_complete[subset, -2:]).T, alpha=0.2, color='black', linestyle='--')

        labeled = False
        for r_idx, r in enumerate(rocs):
            corr_x = x_roc_complete[r_idx]
            # Find matching
            len_r = len(r)
            
            index = 0
            min_dist = 10000000
            for i in range(len(corr_x-len_r)):
                dist = dtw(corr_x[i:i+len_r], r)
                if dist < min_dist:
                    index = i
                    min_dist = dist

            if r_idx == closest_roc:
                ax3.plot(x_roc_complete[r_idx], color='black', alpha=0.5)
                ax3.plot(x_pos_label[0], rocs_complete[r_idx, -2:].T, alpha=0.5, color='black', linestyle='--')
                ax3.plot(list(range(index, index+len_r)), r, alpha=1, color='orange', label='Closest in $R_{' + str(idx_after) + '}$')
            else:
                if r_idx in subset:
                    ax3.plot(list(range(index, index+len_r)), r, alpha=0.3, color='blue')

        ax3.set_xticks([])

        f.legend(bbox_to_anchor=(0.50, 0.92))
        
        ax1.set_xticks(np.arange(0, 16, 3), np.arange(0, 16, 3)[::-1])
        ax2.set_xticks(np.arange(0, 16, 3), np.arange(0, 16, 3)[::-1])
        ax3.set_xticks(np.arange(0, 16, 3), np.arange(0, 16, 3)[::-1])
        ax1.set_xlabel('lag')
        ax2.set_xlabel('lag')
        ax3.set_xlabel('lag')
        
        path = f'plots/drift_model_change_{ds_name}_{ds_index}_{drift_idx}_{data_idx}'
        print(path)
        f.suptitle(f'Change in model selection after drift detection')
        #f.tight_layout()
        f.savefig(path + '.png')
        f.savefig(path + '.pdf')
        #f.savefig('plots/drift_model_change.pdf', bbox_inches='tight')

def extract_rules_rf(x, estimators):
    x = x.reshape(1, -1)
    sample_id = 0

    feature_thresholds = np.zeros((x.shape[-1], 2))
    feature_thresholds[:, 0] = -np.inf
    feature_thresholds[:, 1] = np.inf

    for tree in estimators:

        feature = tree.tree_.feature
        threshold = tree.tree_.threshold

        node_indicator = tree.decision_path(x)
        leave_id = tree.apply(x)
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:node_indicator.indptr[sample_id + 1]]

        for node_id in node_index:
            if leave_id[sample_id] == node_id:
                continue

            if (x[sample_id, feature[node_id]] <= threshold[node_id]):
                old = feature_thresholds[feature[node_id], 1]
                new = min(old, threshold[node_id])
                feature_thresholds[feature[node_id], 1] = new
            else:
                old = feature_thresholds[feature[node_id], 0]
                new = max(old, threshold[node_id])
                feature_thresholds[feature[node_id], 0] = new

    return feature_thresholds

def get_explanation(x, model):
    explainer = TreeExplainer(model)
    return explainer.shap_values(x)

def why_explanation(dataset_index, idx_invest):
    print(dataset_index, idx_invest)

    data = get_all_datasets()[dataset_index:(dataset_index+1)]
    for idx, (ds_name, ds_index) in enumerate(data):
        print(ds_name, ds_index)
        X_train, X_val, X_test = load_data(ds_name, ds_index)
        models = load_models(ds_name, ds_index, None, False)

        X_test_win, y_test_win = windowing(X_test, L=15)

        comp = OS_PGSM_St_Faster(models, lag=15, big_lag=25, threshold=0.01)
        print(X_train.shape, X_val.shape, X_test.shape)
        comp.run(X_train, X_val, X_test)

        x = X_test_win[idx_invest]
        y = y_test_win[idx_invest]
        forecaster_id = comp.test_forecasters[idx_invest]
        forecaster = models[forecaster_id]
        if isinstance(forecaster, GradientBoostingRegressor):
            continue
        forecasted_value = forecaster.predict(x.reshape(1, -1)).squeeze()
        print(forecaster)

        # Find closest roc
        dists = [dtw(x, r) for r in comp.rocs[forecaster_id] if r.shape[0] != 0]
        min_dist_idx = np.argmin(dists)
        closest_roc = comp.rocs[forecaster_id][min_dist_idx]

        # furthest_roc = [dtw(r, x) for r in comp.rocs[f_idx]]
        # furthest_forecaster
        max_dist = 0
        furthest_forecaster_idx = None
        furthest_roc = None
        furthest_roc_idx = None
        for f_idx in range(len(comp.rocs)):
            for roc_idx, r in enumerate(comp.rocs[f_idx]):
                dist = dtw(r, x)
                if dist > max_dist:
                    max_dist = dist
                    furthest_forecaster_idx = f_idx
                    furthest_roc = r
                    furthest_roc_idx = roc_idx


        print(comp.test_forecasters)
        print(forecaster_id, furthest_forecaster_idx)

        forecasted_value_furthest = models[furthest_forecaster_idx].predict(x.reshape(1, -1)).squeeze()

        shaps = get_explanation(x, forecaster)
        shaps = np.abs(shaps)

        if isinstance(forecaster, DecisionTreeRegressor):
            estimators = [forecaster]
        else:
            estimators = forecaster.estimators_
        thresholds = extract_rules_rf(x, estimators)

        subplots = (2, 1)
        fig_width_in, fig_height_in = set_style(subplots, autolayout=False, height_scale=1.4)

        f, (ax1, ax2) = plt.subplots(subplots[0], subplots[1], figsize=(fig_width_in, fig_height_in))
        
        ax1.plot(x, color='black', label='Window to forecast')
        ax1.plot(comp.rocs_complete[forecaster_id][min_dist_idx][:-1], color='black', alpha=0.4)
        ax1.plot(comp.rocs_complete[furthest_forecaster_idx][furthest_roc_idx][:-1], color='black', alpha=0.4)

        # Find start point
        x_roc = comp.rocs_complete[forecaster_id][min_dist_idx][:-1]
        i = 0
        for _i in range(len(x_roc)-len(closest_roc)):
            subseries = x_roc[_i:_i+len(closest_roc)]
            if np.all(closest_roc == subseries):
                break
            i += 1

        ax1.plot(np.arange(len(closest_roc))+i, closest_roc, color='green', label='Closest RoC')

        # Find start point
        x_roc = comp.rocs_complete[furthest_forecaster_idx][furthest_roc_idx][:-1]
        i = 0
        for _i in range(len(x_roc)-len(furthest_roc)):
            subseries = x_roc[_i:_i+len(furthest_roc)]
            if np.all(furthest_roc == subseries):
                break
            i += 1

        ax1.plot(np.arange(len(furthest_roc))+i, furthest_roc, color='red', label='Furthest RoC')

        ax1.set_title('Why was forecaster $f_i$ chosen?')
        ax1.set_xlabel('$t$')

        ax1.legend()

        # ----------------------

        ax2.bar(np.arange(15), shaps, color='blue', alpha=0.2, label='Shapley values\nfor prediction')

        print(thresholds)
        color = 'orange'
        ax2.fill_between(np.arange(len(thresholds)), thresholds[:, 0], thresholds[:, 1], alpha=0.1, color=color)
        for x_idx, (low, high) in enumerate(thresholds):
            if x_idx == 0:
                ax2.vlines(x_idx, low, high, colors=color, alpha=0.8, label='Feature intervals')
            else:
                ax2.vlines(x_idx, low, high, colors=color, alpha=0.8)

        ax2.plot(x, color='black', label='Window to forecast')
        ax2.plot(np.arange(2)+len(x)-1, [x[-1], y], color='blue', label='Ground Truth')
        ax2.plot(np.arange(2)+len(x)-1, [x[-1], forecasted_value], color='green', label='Forecasted value\nClosest RoC')
        ax2.plot(np.arange(2)+len(x)-1, [x[-1], forecasted_value_furthest], color='red', label='Forecasted value\nFurthest RoC')
        ax2.set_title(f'Why did forecaster $f_i$ predict its output?')

        ax1.set_xlabel('lag')
        ax2.set_xlabel('lag')

        ax1.set_xticks(np.arange(0, 15, 2), np.arange(1, 16, 2)[::-1])
        ax2.set_xticks(np.arange(0, 16, 3), np.arange(0, 16, 3)[::-1])

        f.subplots_adjust(bottom=0.2, top=0.95, hspace=0.3)
        ax2.legend(ncol=2, bbox_to_anchor=(0.9, -0.2))

        f.savefig(f'plots/why_{dataset_index}_{idx_invest}.png')
        f.savefig(f'plots/why_{dataset_index}_{idx_invest}.pdf')

def change_feature_importance(ds_name, ds_index):

        X_train, X_val, X_test = load_data(ds_name, ds_index)
        models = load_models(ds_name, ds_index, None, False)

        X_test_win, y_test_win = windowing(X_test, L=15)

        comp = OS_PGSM_Faster(models, lag=15, big_lag=25, threshold=0.01)
        comp.run(X_train, X_val, X_test)

        drifts = comp.drifts_detected
        print(drifts)

        if len(drifts) == 0:
            return

        for drift_idx in range(len(drifts)-1):

            t_drift = drifts[drift_idx]
            for after_idx in range(t_drift, len(X_test_win)):
                x_before = X_test_win[t_drift-1]
                y_before = y_test_win[t_drift-1]
                
                x_after = X_test_win[after_idx]
                y_after = y_test_win[after_idx]

                idx_before = np.argmin([min([dtw(r, x_before) for r in roc]) if len(roc) > 0 else 100000 for roc in comp.roc_history[drift_idx]])
                idx_after = np.argmin([min([dtw(r, x_after) for r in roc]) if len(roc) > 0 else 100000 for roc in comp.roc_history[drift_idx+1]])

                if idx_before == idx_after:
                    return

                m_before = models[idx_before]
                m_after = models[idx_after]

                if not (isinstance(m_before, RandomForestRegressor) and isinstance(m_after, RandomForestRegressor)):
                    return

                pred_before = m_before.predict(x_before.reshape(1, -1)).squeeze()
                pred_after = m_after.predict(x_after.reshape(1, -1)).squeeze()

                l_after_new = (pred_after - y_after)**2
                l_after_old = (m_before.predict(x_after.reshape(1, -1)).squeeze() - y_after)**2
                if l_after_new >= l_after_old:
                    return

                # Better visibility
                shap_scale_factor = 2

                shaps_before = get_explanation(x_before, m_before) * shap_scale_factor
                shaps_after = get_explanation(x_after, m_after) * shap_scale_factor

                subplots = (2, 1)
                fig_width_in, fig_height_in = set_style(subplots)

                f, (ax1, ax2) = plt.subplots(subplots[0], subplots[1], figsize=(fig_width_in, fig_height_in))

                ax1.bar(np.arange(15), np.abs(shaps_before), color='blue', alpha=0.2, label='Shapley values')

                ax2.bar(np.arange(15), np.abs(shaps_after), color='blue', alpha=0.2)
                
                ax1.plot(x_before, color='black')
                ax1.plot([14, 15], [x_before[-1], pred_before], color='red', label='Prediction before')
                ax1.plot([14, 15], [x_before[-1], y_before], color='green', label='Ground Truth')

                ax2.plot(x_after, color='black')
                ax2.plot([14, 15], [x_after[-1], pred_after], color='orange', label='Prediction after')
                ax2.plot([14, 15], [x_after[-1], m_before.predict(x_after.reshape(1, -1))], color='red')
                ax2.plot([14, 15], [x_after[-1], y_after], color='green')

                ax1.set_xticks(np.arange(0, 16, 3), np.arange(0, 16, 3)[::-1])
                ax2.set_xticks(np.arange(0, 16, 3), np.arange(0, 16, 3)[::-1])
                ax1.set_xlabel('lag')
                ax2.set_xlabel('lag')
                ax1.set_title('Feature importance $f_{' + str(idx_before) + '}$ on $x^{(t)}$ before drift')
                ax2.set_title('Feature importance $f_{' + str(idx_after) + '}$ on $x^{(t+' + str(after_idx) + ')}$ after drift')

                #plt.rcParams.update({'figure.subplot.hspace': 0.1})

                # f.legend(bbox_to_anchor=(1.29, 0.75), borderaxespad=0.0)
                # f.tight_layout()
                f.legend(bbox_to_anchor=(0.9, 0.02), ncol=2)
                #f.legend()
                f.savefig(f'plots/change_importance_{ds_name}_{ds_index}_{drift_idx}_{after_idx}.png', bbox_inches='tight')
                f.savefig(f'plots/change_importance_{ds_name}_{ds_index}_{drift_idx}_{after_idx}.pdf', bbox_inches='tight')

def drift_xai():

    candidates = [
        ('pedestrian_counts', 13),
    ]

    def find_best_forecaster(x, rocs):
        best_model = None
        smallest_distance = 1e8
        closest_roc_member = None

        for i in range(len(rocs)):
            x = x.squeeze()
            for r in rocs[i]:
                distance = dtw(r, x)
                if distance < smallest_distance:
                    best_model = i
                    smallest_distance = distance
                    closest_roc_member = r

        return best_model, closest_roc_member

    # data = get_all_datasets()
    # for idx, (ds_name, ds_index) in enumerate(data):
    for (ds_name, ds_index) in candidates:
        X_train, X_val, X_test = load_data(ds_name, ds_index)
        models = load_models(ds_name, ds_index, None, False)

        X_test_win, y_test_win = windowing(X_test, L=15)

        comp = OS_PGSM_Faster(models, lag=15, big_lag=50, threshold=0.01)
        comp.run(X_train, X_val, X_test)

        drifts = comp.drifts_detected
        history = comp.roc_history

        # Find where model changes after drift
        for idx, drift_idx in enumerate(drifts):
            rocs_after = history[idx+1]
            rocs_before = history[idx]

            try:
                x, y = X_test_win[drift_idx], y_test_win[drift_idx]
            except Exception:
                continue
            found_before, closest_before = find_best_forecaster(x, rocs_before)
            found_after, closest_after = find_best_forecaster(x, rocs_after)
            if found_before == found_after or found_before is None or found_after is None or dtw(closest_before, closest_after) <= 0.01:
                continue

            pred_before = models[found_before].predict(x.reshape(1, -1))
            pred_after = models[found_after].predict(x.reshape(1, -1))
            loss_before = (pred_before - y)**2
            loss_after = (pred_after - y)**2

            if loss_after >= loss_before:
                continue

            print(drift_idx, found_before, found_after)
            f, ax = plt.subplots(figsize=(6.4, 2.8))

            ax.plot(x, color='black', label='input')
            ax.plot([14, 15], [x[-1], y[0]], color='black', linestyle='--')
            ax.plot(closest_before, color='red', label='before')
            ax.plot([14, 15], [x[-1], pred_before[0]], color='red', linestyle='--', label='pred before')
            ax.plot(closest_after, color='green', label='after')
            ax.plot([14, 15], [x[-1], pred_after[0]], color='green', linestyle='--', label='pred after')
            ax.set_xlabel('$t$')
            ax.set_title('Closest RoC members before and after concept drift')
            ax.legend()
            f.savefig(f'plots/drift_xai_{ds_name}_#{ds_index}_{drift_idx}.png')
            f.savefig(f'plots/drift_xai_{ds_name}_#{ds_index}_{drift_idx}.pdf')

def roc_best_with_output(ds_name, ds_index, drift_idx):
    def find_unique_rows(X):
        already_seen = []
        to_return = []
        for r_idx, r in enumerate(X):
            if len(already_seen) == 0:
                already_seen.append(r)
                to_return.append(r_idx)
            else:
                if not np.any(np.all(np.vstack(already_seen) == r, axis=1)):
                    already_seen.append(r)
                    to_return.append(r_idx)

        return np.array(to_return)
    
    print(ds_name, ds_index)
    X_train, X_val, X_test = load_data(ds_name, ds_index)
    models = load_models(ds_name, ds_index, None, False)

    X_test_win, y_test_win = windowing(X_test, L=15)

    comp = OS_PGSM_Faster(models, lag=15, big_lag=25, threshold=0.01, min_roc_length=3)
    comp.run(X_train, X_val, X_test)

    drifts = comp.drifts_detected
    print(drifts)
    idx_invest = drifts[drift_idx]

    x = X_test_win[idx_invest]
    y = y_test_win[idx_invest]
    forecaster_id = comp.test_forecasters[idx_invest]
    forecaster = models[forecaster_id]
    forecasted_value = forecaster.predict(x.reshape(1, -1)).squeeze()

    rng = np.random.RandomState(18581)
    rocs = deepcopy(comp.roc_history[drift_idx][forecaster_id])
    x_rocs = np.array(deepcopy(comp.roc_complete_history[drift_idx][forecaster_id]))
    y_rocs = deepcopy(x_rocs[:, -2:])
    x_rocs = x_rocs[:, :-1]

    closest_roc_index = np.argmin([dtw(r, x) for r in rocs])
    closest_roc = rocs[closest_roc_index]
    closest_x_roc = x_rocs[closest_roc_index]
    closest_y_roc = y_rocs[closest_roc_index]

    subset_size = 5
    unique_rows = find_unique_rows(x_rocs)
    subset = rng.choice(unique_rows, size=min(len(unique_rows), subset_size), replace=False)

    x_rocs = x_rocs[subset]
    y_rocs = y_rocs[subset]
    rocs = [rocs[i] for i in subset]

    subplots = (2, 1)
    fig_width_in, fig_height_in = set_style(subplots)

    f, (ax1, ax2) = plt.subplots(subplots[0], subplots[1], figsize=(fig_width_in, fig_height_in))
    ax1.plot(x, color='black')
    ax1.plot([14, 15], [x[-1], y], color='green', label='Ground Truth')
    ax1.plot([14, 15], [x[-1], forecasted_value], color='red', label='Prediction')

    ax2.plot(x_rocs.T, color='black', alpha=0.2)
    ax2.plot(np.array([[14, 15]]).repeat(len(y_rocs), 0).T, y_rocs.T, color='black', linestyle='--', alpha=0.7)
    print(closest_y_roc)
    ax2.plot([14, 15], closest_y_roc, color='black', linestyle='--', alpha=0.7)

    # Find start point
    for x_idx in range(len(x_rocs)):
        x_roc = x_rocs[x_idx]
        r = rocs[x_idx]
        i = 0
        for _i in range(len(x_roc)-len(r)):
            subseries = x_roc[_i:_i+len(r)]
            if np.all(r == subseries):
                break

            i += 1

        ax2.plot(np.arange(len(r))+i, r, color='blue', alpha=0.5)

    ax2.plot(closest_x_roc, color='black', alpha=0.7)
    # Find start point
    i = 0
    for _i in range(len(closest_x_roc)-len(closest_roc)):
        subseries = closest_x_roc[_i:_i+len(closest_roc)]
        if np.all(closest_roc == subseries):
            break
        i+=1
    ax2.plot(np.arange(len(closest_roc))+i, closest_roc, color='orange', alpha=1, label='Closest RoC\nmember')

    ax1.set_xticks(np.arange(0, 16, 3), np.arange(0, 16, 3)[::-1])
    ax2.set_xticks(np.arange(0, 16, 3), np.arange(0, 16, 3)[::-1])
    ax1.set_xlabel('lag')
    ax2.set_xlabel('lag')
    f.suptitle('Visualization of RoC for model $f_{' + str(forecaster_id) + '}$')
    f.legend(bbox_to_anchor=(0.9, 0.45))
    f.savefig(f'plots/rocs_with_prediction_{ds_name}_{ds_index}_{drift_idx}.png')
    f.savefig(f'plots/rocs_with_prediction_{ds_name}_{ds_index}_{drift_idx}.pdf')

def main():

    #### Figure 1
    #why_explanation(513, 77)
    #why_explanation(516, 4)
    #why_explanation(185, 77)
    why_explanation(128, 88)


    #### Figure 2
    change_feature_importance('m4_weekly', 82)

    #### Figure 3
    roc_best_with_output('m4_hourly', 16, 0)

    #### Figure 4
    drift_model_no_change('m4_hourly', 4)

    #### Figure 5
    drift_model_change()

if __name__ == '__main__':
    main()
