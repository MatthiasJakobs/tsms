import pandas as pd
import numpy as np

from datasets.dataloading import get_all_datasets

selectors = [
    'TShapSelector',
    'TShapSelector-dtw',
]

def main():
    # How often DTW is better than Euclidean
    wins, losses = 0, 0
    for ds_name, ds_index in get_all_datasets():
        df = pd.read_csv(f'test_predictors/predictors_{ds_name}_#{ds_index}.csv', index_col=0, header=0)
        ground_truth = df['best-case'].to_numpy()

        comparison = df['TShapSelector'].to_numpy()
        accuracy_real = np.mean(ground_truth == comparison)
        comparison = df['TShapSelector-dtw'].to_numpy()
        accuracy_dtw_real = np.mean(ground_truth == comparison)

        if accuracy_dtw_real > accuracy_real:
            wins += 1
        elif accuracy_dtw_real < accuracy_real:
            losses += 1

    print('Wins of DTW over Euclidean')
    print(f'{wins} wins, {losses} losses, {len(get_all_datasets()) - wins - losses} draws')

    # Takeaway: Even though Euclidean selects a bit more the best possible model, 
    # the other selections seem to reduce the performance more.
    # Meaning: If both choose not the best, DTW chooses better on average

if __name__ == '__main__':
    main()
