import numpy as np
import pandas as pd
from os.path import exists, join
from os import makedirs

from datasets.dataloading import get_all_datasets
from critdd import Diagram
from sklearn.metrics import mean_squared_error as mse
from experiments import results_path
from single_models import get_single_models

def create_cd_diagram(compositors_to_check, output_name, title, treatment_mapping=None):

    used_datasets = 0
    total_datasets = len(get_all_datasets())

    losses = []

    for ds_name, ds_index in get_all_datasets():
        ds_path = join(results_path, f'test_{ds_name}_#{str(ds_index)}.csv')
        if exists(ds_path):
            df = pd.read_csv(ds_path)
            if all([comp_name in df.columns for comp_name in compositors_to_check]):
                
                scores = []
                y = df['y'].to_numpy().squeeze()
                for comp_name in compositors_to_check:
                    pred = df[comp_name].to_numpy().squeeze()
                    scores.append(mse(y, pred, squared=False)) # RMSE

                losses.append(np.array(scores))
                used_datasets += 1

    losses = np.vstack(losses)

    if treatment_mapping is not None:
        treatment_names = [treatment_mapping[name] for name in compositors_to_check]
    else:
        treatment_names = compositors_to_check

    print(f'Used {used_datasets} datasets from {total_datasets}')

    diagram = Diagram(
        losses,
        treatment_names=treatment_names,
        maximize_outcome=False
    )

    diagram.to_file(output_name, title=title)

def main():
    print(results_path)
    # First: Ablation
    #compositors_to_check = ['SAX-dep-ST-pred', 'SAX-indep-ST-pred', 'KS-ST', 'KS-ST-sax-perturbations', 'SAX-dep-ST', 'SAX-indep-ST']
    makedirs('plots', exist_ok=True)

    # Load all single model names 
    single_model_names = [m[0] for m in get_single_models()]

    compositors_to_check = [
        'SAX-dep-ST-onlybest',
        'SAX-dep-ST-pred-onlybest',
        'SAX-indep-ST-onlybest',
        'SAX-indep-ST-pred-onlybest',
        'KS-ST-onlybest'
    ] + single_model_names
    print(compositors_to_check)

    output_name = 'plots/cd-ablation.tex'
    title = 'Ablation'
    treatment_mapping = {
        k: k.replace('_', '-') for k in compositors_to_check
    }
    create_cd_diagram(compositors_to_check, output_name, title, treatment_mapping=treatment_mapping)

    exit()

    # Second: Best vs. previous
    compositors_to_check = ['KS-ST-onlybset', 'OS-PGSM-ST-onlybest', 'OEP-ROC-ST-redo', 'KS-ST', 'OEP-ROC-15', 'OS-PGSM', 'OS-PGSM-ST', 'SAX-indep-ST-pred', 'SAX-indep-ST-pred-onlybest']
    output_name = 'plots/cd-comparison.tex'
    title = 'Best vs. previous'
    create_cd_diagram(compositors_to_check, output_name, title)

if __name__ == '__main__':
    main()
