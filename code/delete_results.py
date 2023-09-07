import argparse
import pandas as pd

from os.path import exists

from datasets.dataloading import get_all_datasets
from experiments import results_path

def delete(files, experiment_names):

    print('The following actions will be taken:')
    for file in files:
        if not exists(file):
            continue
        df = pd.read_csv(file, header=0)
        for exp_name in experiment_names:
            if exp_name in df.columns:
                print('DELETE', exp_name, 'from', file)

    choice = input('Continue? [y/n] ')
    if choice not in ['y', 'Y', 'n', 'N']:
        raise RuntimeError('Unknown input choice', choice)

    if choice in ['n', 'N']:
        print('Abort...')
        exit()

    for file in files:
        if not exists(file):
            continue
        df = pd.read_csv(file, header=0)
        df = df.drop(columns=experiment_names, errors='ignore')
        df.to_csv(file, index=None)

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', nargs='+')
    parser.add_argument('-e', '--experiments', nargs='+', required=True)
    args = parser.parse_args()

    if args.files is None:
        print(results_path)
        files = [f'{results_path}/test_{t[0]}_#{t[1]}.csv' for t in get_all_datasets()]
    else:
        files = args.files

    delete(files, args.experiments)
