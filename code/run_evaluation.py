import numpy as np
import pandas as pd

from os.path import exists, join
from sklearn.metrics import mean_squared_error as mse

from datasets.dataloading import get_all_datasets
from experiments import results_path

def scale_loss(L, L_min, L_max):
    return (L - L_min) / (L_max - L_min)

def main():

    data = get_all_datasets()
    predictor_names = sorted(pd.read_csv('results/test_electricity_hourly_#0.csv', index_col=0).drop(columns=['y', 'best-case-real', 'worst-case-real']).columns.tolist())
    print(predictor_names)
    n_predictors = len(predictor_names)
    n_datasets = len(data)

    norm_loss_matrix = np.zeros((n_datasets, n_predictors))

    for row_idx, (ds_name, ds_index) in enumerate(data):
        ds_path = join(results_path, f'test_{ds_name}_#{str(ds_index)}.csv')
        df_test = pd.read_csv(ds_path, index_col=0)

        # Establish upper and lower loss bounds with worst/bestcase selection
        y = df_test['y'].to_numpy().squeeze()
        best_case_prediction = df_test['best-case-real'].to_numpy().squeeze()
        worst_case_prediction = df_test['worst-case-real'].to_numpy().squeeze()
        L_min = mse(y, best_case_prediction, squared=False)
        L_max = mse(y, worst_case_prediction, squared=False)

        for col_idx, model_name in enumerate(predictor_names):
            pred = df_test[model_name].to_numpy().squeeze()
            L = mse(y, pred, squared=False)
            L_scaled = scale_loss(L, L_min, L_max)
            assert L_scaled >= 0 and L_scaled <= 1, model_name

            norm_loss_matrix[row_idx, col_idx] = L_scaled

    mean_losses = np.mean(norm_loss_matrix, axis=0)
    loss_ranking = np.argsort(-mean_losses)
    for m_name, s_loss in zip(np.array(predictor_names)[loss_ranking], mean_losses[loss_ranking]):
        print(m_name, s_loss)

    print('---------------------------------------')

    loss_ranking = np.argsort(np.argsort(norm_loss_matrix, axis=1), axis=1)
    mean_losses = np.mean(loss_ranking, axis=0)

    sorting = np.argsort(-mean_losses)
    for m_name, m_mean_rank in zip(np.array(predictor_names)[sorting], mean_losses[sorting]):
        print(m_name, m_mean_rank)


    



if __name__ == '__main__':
    main()