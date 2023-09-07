# Run all experiments

from train_single_models import train_and_evaluate
from run_prediction import run_compositors, print_predictors
from plotting import create_cd_diagram, create_cd_best_overall, create_single_prediction_files, create_cd_diagram_compositors

if __name__ == '__main__':

    # Single models
    train_and_evaluate(None)

    # Compositors
    override = False
    debug = False
    dry_run = False
    run_compositors(None, override=override, noparallel=debug, dry_run=dry_run)

    # Results
    create_cd_diagram_compositors()

    # Export
    create_single_prediction_files()

