import numpy as np
import pandas as pd
import glob

from os.path import join, exists
from sklearn.metrics import mean_squared_error as mse
from os import makedirs

from datasets.dataloading import get_all_datasets
from utils import get_sax_postfix
from single_models import get_sax_postfix, get_single_models
from train_single_models import load_data
from experiments import results_path

postfixes_to_plot = ['real', 'quant3', 'quant5', 'quant7', 'quant12', 'quant25']

output_mapping = {}

def create_cd_diagram(sax_alphabet_size):
    from critdd import Diagram

    postfix = get_sax_postfix(sax_alphabet_size)
    used_datasets = 0
    total_datasets = len(get_all_datasets())


    for split in ['val', 'test']:
        losses = []
        model_names, _ = get_single_models()
        model_names = [name + '-' + postfix for name in model_names]

        for ds_name, ds_index in get_all_datasets():
            ds_path = join(results_path, f'{split}_{ds_name}_#{str(ds_index)}.csv')
            if exists(ds_path):
                df = pd.read_csv(ds_path)
                y = df['y'].to_numpy().squeeze()
                scores = []

                for model_name in model_names:
                    pred = df[model_name].to_numpy().squeeze()
                    scores.append(mse(y, pred, squared=False)) # RMSE

                losses.append(np.array(scores))
                used_datasets += 1

        model_names = [x.replace('_', '-') for x in model_names]
        model_names = [x.replace('-real', '') for x in model_names]
        losses = np.vstack(losses)

        for ds_idx, ds_loss_row in enumerate(losses):
            for l_idx, l in enumerate(ds_loss_row):
                equals = np.where(l == ds_loss_row)[0]
                if not (len(equals) == 1 and equals[0] == l_idx):
                    print(ds_idx, l_idx, equals)

        print(f'Used {used_datasets} datasets from {total_datasets}')

        diagram = Diagram(
            losses,
            treatment_names=model_names,
            maximize_outcome=False
        )

        diagram.to_file(f'plots/cd-single-models-{split}-{postfix}.tex', title=f'Single model performances {split} {postfix}')

def create_cd_best_overall():
    from critdd import Diagrams

    model_names, _ = get_single_models()
    diagram_names = postfixes_to_plot

    critt_df = pd.DataFrame()

    # Create pandas dataframe to send to critd

    for ds_name, ds_index in get_all_datasets():
        ds_path = join(results_path, f'test_{ds_name}_#{str(ds_index)}.csv')
        full_ds_name = f'{ds_name}-{ds_index}'

        if exists(ds_path):
            df = pd.read_csv(ds_path)
            y = df['y'].to_numpy().squeeze()


            for postfix in postfixes_to_plot:
                for model_name in model_names:
                    pred = df[model_name + '-' + postfix].to_numpy().squeeze()
                    loss = mse(y, pred, squared=False) # RMSE
                    critt_df = critt_df.append({'dataset': full_ds_name, 'method': model_name, 'loss': loss, 'diagram': postfix }, ignore_index=True)

    print(critt_df)
    # construct a sequence of CD diagrams
    treatment_names = critt_df["method"].unique()
    diagram_names = critt_df["diagram"].unique()
    Xs = [] # collect an (n,k)-shaped matrix for each diagram
    for n in diagram_names:
        diagram_df = critt_df[critt_df.diagram == n].pivot(
            index = "dataset",
            columns = "method",
            values = "loss"
        )[treatment_names] # ensure a fixed order of treatments
        Xs.append(diagram_df.to_numpy())
    two_dimensional_diagram = Diagrams(
        np.stack(Xs),
        diagram_names = diagram_names,
        treatment_names = treatment_names,
        maximize_outcome = False
    )

    two_dimensional_diagram.to_file(
        "plots/2d_example.tex",
    )

    # Print top 5 models overall (in terms of mean loss over all datasets)
    model_wins = {}
    for ds_name, ds_index in get_all_datasets():
        full_ds_name = f'{ds_name}-{ds_index}'
        df = critt_df[critt_df['dataset'] == full_ds_name]
        df['full_model_name'] = df['method'] + '-' + df['diagram']
        best_model_index = np.argmin(df['loss'])
        best_model_name = df.iloc[best_model_index].full_model_name
        try:
            model_wins[best_model_name] += 1
        except KeyError:
            model_wins[best_model_name] = 1

    print(model_wins)

def create_single_prediction_files():

    model_names, _ = get_single_models()
    included_selections = [
        'best-case', 
        'worst-case', 
        'TSMS',
        'TSMS-St',
        'TSMS-Per',
    ]
    postfix = 'real'

    makedirs(f'single_predictions/', exist_ok=True)
    makedirs(f'train_data_csv/', exist_ok=True)
    for ds_name, ds_index in get_all_datasets():
        # Save train data
        X_train, _, _ = load_data(ds_name, ds_index)
        df = pd.DataFrame()
        df['train_data'] = X_train
        df.to_csv(f'train_data_csv/{ds_name}_#{ds_index}.csv')

        # Save predictions
        for split in ['val', 'test']:
            new_df = pd.DataFrame()
            ds_path = join(results_path, f'{split}_{ds_name}_#{str(ds_index)}.csv')

            if exists(ds_path):
                df = pd.read_csv(ds_path)
                y = df['y'].to_numpy().squeeze()
                new_df['y'] = y

                for model_name in model_names:
                    full_model_name = model_name + '-' + postfix
                    pred = df[full_model_name].to_numpy().squeeze()
                    full_model_name = output_mapping.get(full_model_name, full_model_name)
                    new_df[full_model_name] = pred

                if split == 'test':
                    for selector_name in included_selections:
                        full_selector_name = selector_name + '-' + postfix
                        pred = df[full_selector_name].to_numpy().squeeze()
                        full_selector_name = output_mapping.get(full_selector_name, full_selector_name)
                        new_df[full_selector_name] = pred

            new_ds_path = join(f'single_predictions/{split}_{ds_name}_#{str(ds_index)}.csv')
            new_df.to_csv(new_ds_path)

def create_cd_diagram_compositors():
    from critdd import Diagram

    used_datasets = 0
    makedirs('plots', exist_ok=True)

    losses = []
    single_names, _ = get_single_models()

    selector_names = [
        'best-case-real',
        'worst-case-real',
        'TSMS-real',
        'TSMS-St-real',
        'TSMS-Per-real',
    ]

    selector_names += [sn + '-real' for sn in single_names]

    for ds_name, ds_index in get_all_datasets():
        ds_path = join(results_path, f'test_{ds_name}_#{str(ds_index)}.csv')
        if exists(ds_path):
            df = pd.read_csv(ds_path)
            y = df['y'].to_numpy().squeeze()
            scores = []

            for model_name in selector_names:
                pred = df[model_name].to_numpy().squeeze()
                scores.append(mse(y, pred, squared=False)) # RMSE

            losses.append(np.array(scores))
            used_datasets += 1

    losses = np.vstack(losses)

    selector_names = [output_mapping.get(name, name) for name in selector_names]
    selector_names = [name.replace('-real', '') for name in selector_names]

    diagram = Diagram(
        losses,
        treatment_names=selector_names,
        maximize_outcome=False,
    )

    # Sorting average_ranks
    argsort_ranks = np.argsort(diagram.average_ranks)
    sorted_ranks = diagram.average_ranks[argsort_ranks]
    sorted_names = [selector_names[i] for i in argsort_ranks]

    for name, avg_rank in list(zip(sorted_names, sorted_ranks))[::-1]:
        print(name, avg_rank)
    diagram.to_file('plots/cd-ablation.tex', axis_options={'title': 'Ablation'})

# Latex table export
def export_table(array, path):

    def add_space(x):
        return x + '~~~~'

    map_names = {
        'pred.es': 'ETS',
        'pred.a': 'ARIMA',
        'pred.gbm': 'GBM',
        'pred.rf': 'RF',
        'pred.knn': 'KNN-RoC',
        'pred.cnn': 'CNN',
        'pred.dets.single': 'DETS',
        'pred.ade.single': 'ADE',
        'pred.cnn.lstm10': 'CNN-LSTM',
        'rf-8-real': 'Best-Single',
        'TSMS-Per-real': '\\textbf{TSMS-Per}',
        'TSMS-St-real': '\\textbf{TSMS-St}',
        'TSMS-real': '\\textbf{TSMS}'
    }

    df = pd.DataFrame()
    for idx, (name, mu_rank, var_rank) in enumerate(array):
        new_df = pd.DataFrame({'Method': [map_names[name]], 'Avg. Rank ($\pm$ variance)': [f'${mu_rank:.2f} \pm {var_rank:.2f}$']})
        df = pd.concat([df, new_df], ignore_index=True)

    df.T.to_latex(path, index=False, escape=False)
    print(df)

def create_comparison_graphic():
    ### --- Find best single predictor ---
    single_names = pd.read_csv('single_predictions/val_electricity_hourly_#0.csv', index_col=0).drop(columns=['y']).columns.tolist()
    compositor_models = ['TSMS-real', 'TSMS-St-real', 'TSMS-Per-real']
    #baseline_models = ['pred.lstm10', 'pred.cnn.lstm10', 'pred.cnn', 'pred.knn', 'pred.rf', 'pred.dets.single', 'pred.ade.single', 'pred.gbm', 'pred.a', 'pred.es']
    baseline_models = ['pred.cnn.lstm10', 'pred.cnn', 'pred.knn', 'pred.dets.single', 'pred.ade.single', 'pred.a', 'pred.es']

    test_file_paths = sorted(glob.glob('single_predictions/test_*'))
    data = get_all_datasets()

    single_ranking = np.zeros((len(single_names)))
    for ds_name, ds_index in data:
        ds_path = join(results_path, f'test_{ds_name}_#{str(ds_index)}.csv')
        df_test = pd.read_csv(ds_path, index_col=0)
        y = df_test['y']
        scores = []
        for single_model_name in single_names:
            pred = df_test[single_model_name].to_numpy().squeeze()
            scores.append(mse(y, pred, squared=False)) # RMSE

        single_ranking += np.array(np.argsort(np.argsort(scores)))
        
    single_ranking /= len(data)
    single_best_model =  single_names[np.argmin(single_ranking)]
    print(single_best_model)

    all_model_names = np.array(compositor_models + baseline_models + [single_best_model])
    losses = np.zeros((len(test_file_paths), len(all_model_names)))
    #ranks = np.zeros((len(test_file_paths), len(all_model_names)))
    for idx, ds_path in enumerate(test_file_paths):
        df_test = pd.read_csv(ds_path, index_col=0)
        y = df_test['y']
        scores = []

        ### Our methods
        for compositor_name in compositor_models:
            pred = df_test[compositor_name].to_numpy().squeeze()
            scores.append(mse(y, pred, squared=False)) # RMSE

        ### Baselines
        for bl_name in baseline_models:
            if bl_name == 'pred.a' or bl_name == 'pred.es':
                bl_file_path = f'results-sax23-4/name.ds{idx+1}.csv'
                df_baselines = pd.read_csv(bl_file_path)
            else:
                bl_file_path = f'results-sax23-3/name.ds{idx+1}.csv'
                df_baselines = pd.read_csv(bl_file_path)

                bl_y = df_baselines['target']
                assert np.all(np.isclose(y - bl_y, np.zeros_like(y)))

            pred = df_baselines[bl_name].to_numpy().squeeze()
            scores.append(mse(y, pred, squared=False)) # RMSE

        ### Best single
        pred = df_test[single_best_model]
        scores.append(mse(y, pred, squared=False)) # RMSE

        #ranks[idx] = np.argsort(np.argsort(scores))
        
        # Normalize losses
        best_loss = mse(y, df_test['best-case-real'], squared=False)
        worst_loss = mse(y, df_test['worst-case-real'], squared=False)
        losses[idx] = (np.array(scores) - best_loss) / (worst_loss - best_loss)

    ranks = np.argsort(np.argsort(losses, axis=1), axis=1)+1
    mean_rank = np.mean(ranks, axis=0)
    var_rank = np.var(ranks, axis=0)

    sorted_indices = np.argsort(-mean_rank)

    to_export = []
    for name, mu_rank, var_rank in zip(all_model_names[sorted_indices], mean_rank[sorted_indices], var_rank[sorted_indices]):
        to_export.append([name, mu_rank, var_rank])

    export_table(to_export, 'plots/ranking_table.tex')

    from critdd import Diagram
    diagram = Diagram(
        losses,
        treatment_names=all_model_names,
        maximize_outcome=False,
    )
    diagram.to_file('plots/main_comp.pdf', axis_options={'title': 'Rank comparison'})

def parse_result_table():
    path = 'tsms-results.csv'
    df = pd.read_csv(path, header=0, delimiter=';')
    #print(df.to_latex())
    print('before')
    print(df)

    print('after')
    df['Wins'] = df['Wins'] + ' (' + df['WSign.'] + ')'
    df['Losses'] = df['Losses'] + ' (' + df['LSign.'] + ')'
    df = df.drop(columns=['WSign.', 'LSign.'])
    print(df.to_latex(float_format='%.2f', index=False))

def main():
    parse_result_table()

if __name__ == '__main__':
    main()
