import numpy as np
import pandas as pd
import argparse
import tqdm

from warnings import simplefilter
from joblib import Parallel, delayed
from os.path import join, exists
from os import makedirs
from sklearn.metrics import mean_squared_error as mse
from tsx.distances import dtw

from experiments import results_path, lag
from single_models import load_models
from datasets.dataloading import get_all_datasets
from train_single_models import load_data
from utils import get_sax_postfix
from model_selection import Ensemble, BestCaseSelector, WorstCaseSelector, TShapSelector, TShapSelectorDriftaware, RoCBaseline 
from ospgsm import OS_PGSM_St_Faster as OSPGSM_ST
from ospgsm import OS_PGSM_Faster as OSPGSM
from ospgsm import OS_PGSM_Periodic as OSPGSM_PER

model_selectors = {
    'best-case': lambda models, sax_alphabet_size: BestCaseSelector(models, sax_alphabet_size),
    'worst-case': lambda models, sax_alphabet_size: WorstCaseSelector(models, sax_alphabet_size),
    # 'RoCBaseline-euc': lambda models, sax_alphabet_size: RoCBaseline(models, sax_alphabet_size, random_state=1234),
    # 'RoCBaseline-dtw': lambda models, sax_alphabet_size: RoCBaseline(models, sax_alphabet_size, random_state=1234, dist_fn=dtw),
    # 'Ens': lambda models, _: Ensemble(models),
    'TSMS': lambda models, sax_alphabet_size: OSPGSM(models, lag=lag, big_lag=25, threshold=0.01, min_roc_length=3),
    'TSMS-St': lambda models, sax_alphabet_size: OSPGSM_ST(models, lag=lag, big_lag=25, threshold=0.01, min_roc_length=3),
    'TSMS-Per': lambda models, sax_alphabet_size: OSPGSM_PER(models, lag=lag, big_lag=25, threshold=0.01, min_roc_length=3),
}

def print_predictors():
    for (ds_name, ds_index) in get_all_datasets():
        predictors = pd.read_csv(f'test_predictors/predictors_{ds_name}_#{ds_index}.csv', header=0, index_col=0)
        print(ds_name, ds_index)
        for predictor_name in predictors.columns:
            print(predictor_name, np.unique(predictors[predictor_name].to_numpy(), return_counts=True))
        print('-------------------------------------')

def run_compositors(sax_alphabet_size, dry_run=False, override=False, noparallel=False):

    def _run_parallel(model_select_name, ds_name, ds_index, sax_alphabet_size, override, dry_run):

        # Load output csv file
        csv_path = join(results_path, f'test_{ds_name}_#{ds_index}.csv')
        prediction_csv = pd.read_csv(csv_path, header=0, index_col=0)

        postfix = get_sax_postfix(sax_alphabet_size)

        run_method = (model_select_name + '-' + postfix not in prediction_csv.columns) or override

        # Quit out early if no need to run
        if not run_method:
            return

        #print (f'Start {model_select_name}-{postfix} on {ds_name}-#{ds_index}')

        X_train, X_val, X_test = load_data(ds_name, ds_index)

        # Load correct models
        models, _ = load_models(ds_name, ds_index, sax_alphabet_size=sax_alphabet_size, return_names=True)

        # Instantiate selector method
        selector = model_selectors[model_select_name](models, sax_alphabet_size)
        preds = selector.run(X_train, X_val, X_test).squeeze()

        prediction_csv[model_select_name + '-' + postfix] = preds

        # chosen_predictor_path = f'test_predictors/predictors_{ds_name}_#{ds_index}.csv'
        # if not exists(chosen_predictor_path):
        #     df_predictors = pd.DataFrame()
        # else:
        #     df_predictors = pd.read_csv(chosen_predictor_path, header=0, index_col=0)

        # df_predictors[model_select_name] = np.array(selector.test_predictors)

        if not dry_run:
            #df_predictors.to_csv(chosen_predictor_path)
            prediction_csv.to_csv(csv_path)

    postfix = get_sax_postfix(sax_alphabet_size)
    makedirs('test_predictors', exist_ok=True)
    # Start everything in parallel 
    if noparallel:
        for model_selector_name in model_selectors.keys():
            for (ds_name, ds_index) in get_all_datasets():
                print(ds_name, ds_index)
                _run_parallel(model_selector_name, ds_name, ds_index, sax_alphabet_size, override, dry_run)
    else:
        n_jobs = -1
        for model_selector_name in model_selectors.keys():
            Parallel(n_jobs=n_jobs, backend="loky")(delayed(_run_parallel)(model_selector_name, ds_name, ds_index, sax_alphabet_size, override, dry_run) for (ds_name, ds_index) in tqdm.tqdm(get_all_datasets(), desc=f'{model_selector_name}-{postfix}'))

def main(dry_run, override, noparallel):

    simplefilter(action="ignore", category=UserWarning)
    print(results_path)

    run_compositors(None, dry_run, override, noparallel)
    run_compositors(3, dry_run, override, noparallel)
    run_compositors(5, dry_run, override, noparallel)
    run_compositors(7, dry_run, override, noparallel)
    run_compositors(12, dry_run, override, noparallel)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--noparallel", action="store_true")
    args = parser.parse_args()
    
    main(dry_run=args.dry_run, override=args.override, noparallel=args.noparallel)
