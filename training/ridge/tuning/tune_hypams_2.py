'''
Tune hypams for ridge with the new setup

we now have a clear and clean global train-test partition of 174/76

within the 176, we have target-specific train-test splits of about 60/40

we find optimal hypams like this:

pick a target e.g. base_load
and params for this model

train a model with these params on 60, cost function: rsme

test model on 40, cost function: total cost after a simulation,
where the other predictors are set to oracle.

'''
import itertools
import itertools
import sys
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))
from training.split.clean_split import PARTITIONS
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import root_mean_squared_error
from src.simulation.simulation import Simulation
from training._features.base_load_features import get_base_load_features
from src.simulation.controllers.mpc.predictors.ml.model_config import MODEL_FEATURES_BY_FAMILY, MODEL_TARGETS


CSV_REPORT_PATH = Path(repo_root) / "training" / "ridge" / "tuning" / "tuning_results.csv"

HYPAM_GRID = {
    "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
}
    
def _build_param_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*grid.values())]


def _build_model_for_target(target: str, params: dict):
    if target in ("base_load", "pv_gen"):
        estimator = Ridge(**params)
    else:
        estimator = RidgeClassifier(**params)

    return Pipeline( # this is the model saved to .pkl and later used in runtime
        steps=[
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )

def tune_single_model(grid, target="base_load"):
    param_configs = _build_param_grid(grid)

    train_ids = PARTITIONS["inner"][target]["train"]
    test_ids = PARTITIONS["inner"][target]["test"]

    if target == "base_load":
        train_df = get_base_load_features(train_ids)
        test_df = get_base_load_features(test_ids)

    for params in param_configs:
        model = _build_model_for_target(target, params)
        feature_columns = MODEL_FEATURES_BY_FAMILY["ridge"][target] # where to look for features (X) in the df
        y_col = "next_value" # where to find y in the df

        X_train, y_train = train_df[feature_columns], train_df[y_col]
        X_test, y_test = test_df[feature_columns], test_df[y_col]
        model.fit(X_train, y_train)

        if target in ("base_load", "pv_gen"):
            return float(root_mean_squared_error(y_test, model.predict(X_test)))

        accuracy = float(model.score(X_test, y_test))
        return accuracy



def tune_all(targets, grid):
    for target in targets:
        print(f"\n=== Tuning {target} ===")
        single_model_df = tune_single_model(grid, target=target)



if __name__ == '__main__':
    tune_all(MODEL_TARGETS, HYPAM_GRID)