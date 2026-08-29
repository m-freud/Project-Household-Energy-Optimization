'''
here we try to find good hypams for any target/model
for training on portugal data
(ch data elsewhere)

procedure:
- train on inner train partition (yes, full partition)
- validate on inner validation partition
- save hypams/score to csv

'''
# paste this to enable src. imports
import itertools
from pathlib import Path
import sys

from training.features.base_load_features import get_base_load_features
from training.features.ev_status_features import get_ev_status_features
from training.features.pv_gen_features import get_pv_gen_features
# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))
from simulation.controllers.mpc.predictors.ml.model_config import MODEL_FEATURES_BY_FAMILY
from src.simulation.run_context import RunContext # noqa
from src.sqlite_connection import sqlite_conn # noqa

from training.split.clean_split import PARTITIONS # noqa

import argparse  # noqa
import pandas as pd  # noqa

GRID_MAP = {
    "ridge": {
        "grid_1": {
            "alpha": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0],
        },
    },
    "xgboost": {
        "grid_1": {
            "n_estimators": [100, 200, 600],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
        },
    },
}

def build_param_grid(self) -> list[dict]:
    keys = list(self.grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*self.grid.values())]


def get_train_test_frames(target, model_family) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
        train_ids = PARTITIONS["inner"][target]["train"]
        test_ids = PARTITIONS["inner"][target]["test"]
        
        if target == "base_load":
            train_df = get_base_load_features(train_ids)
            test_df = get_base_load_features(test_ids)
            y_col = "next_value"
        elif target == "pv_gen":
            train_df = get_pv_gen_features(train_ids)
            test_df = get_pv_gen_features(test_ids)
            y_col = "next_value"
        elif target in ("ev1_status", "ev2_status"):
            train_df = get_ev_status_features(train_ids)
            test_df = get_ev_status_features(test_ids)
            y_col = "next_state"
        else:
            raise ValueError(f"Unknown target: {target}")

        if "ev" in target:
            feature_columns = MODEL_FEATURES_BY_FAMILY[model_family]["ev_status"]
        else:
            feature_columns = MODEL_FEATURES_BY_FAMILY[model_family][target]
        return train_df, test_df, feature_columns, y_col

def hypam_sweep(target: str, model: str, grid: str):
    '''do a hypam sweep by training models for all grid configs and saving the results to a csv file'''

    train_df, test_df, feature_columns, y_col = get_train_test_frames(target=target, model_family=model)

    for param_set in build_param_grid(GRID_MAP[model][grid]):

    print(f"Running hyperparameter sweep for target: {target}, model: {model}, grid: {grid}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        default="pv",
        help="target. choose between base_load,pv,ev1_state,ev2_state",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        help="inference model. choose between xgboost, random_forest, ridge",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="grid_1",
        help="grid",
    )
    args = parser.parse_args()

    hypam_sweep(target=args.target, model=args.model, grid=args.grid)



