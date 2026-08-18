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

Tune hyperparameters for the ridge models using the new clean split setup.

This module is intentionally small and CSV-oriented: it produces a compact
summary table describing how each candidate parameter set performs on the
validation split, and it leaves the full simulation-based cost evaluation behind
an explicit placeholder hook.
'''
import itertools
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.controllers.mpc.predictors.ml.model_config import MODEL_FEATURES_BY_FAMILY, MODEL_TARGETS
from training._features.base_load_features import get_base_load_features
from training._features.ev_status_features import get_ev_status_features
from training._features.pv_gen_features import get_pv_gen_features
from training.split.clean_split import PARTITIONS

CSV_REPORT_PATH = Path(repo_root) / "training" / "ridge" / "tuning" / "tuning_results123.csv"

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


def _score_next_value(model, X_test: pd.DataFrame, y_test: pd.Series, target: str) -> float:
    """Return the direct model score for the next-value target."""
    if target in ("base_load", "pv_gen"):
        return float(root_mean_squared_error(y_test, model.predict(X_test)))

    accuracy = float(model.score(X_test, y_test))
    return float(1.0 - accuracy)


def _score_sim_total_cost_for_target(target: str, params: dict, test_households: list[int]) -> float:
    """Placeholder for the simulation-based total-cost evaluation.

    This function defines the contract we eventually want for tuning: train a
    candidate model for the given target, replace the other predictors with oracle
    values, run the full simulation over the provided test households, and return
    the resulting total cost.

    The modular controller abstraction that performs this simulation is not in
    place yet, so this placeholder intentionally returns a missing value for now.
    """
    del target, params, test_households
    return float("nan")


def _train_test_frames_for_target(target: str, train_ids: list[int], test_ids: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
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
        feature_columns = MODEL_FEATURES_BY_FAMILY["ridge"]["ev_status"]
    else:
        feature_columns = MODEL_FEATURES_BY_FAMILY["ridge"][target]
    return train_df, test_df, feature_columns, y_col


def tune_single_target(target: str, grid: dict, *, partitions: dict | None = None) -> pd.DataFrame:
    """Return one CSV-ready row per candidate parameter set for a single target.

    Output columns are:
    - target
    - params
    - score_next_value
    - score_sim_total_cost
    """
    param_configs = _build_param_grid(grid)
    partition_map = partitions or PARTITIONS
    train_ids = partition_map["inner"][target]["train"]
    test_ids = partition_map["inner"][target]["test"]

    train_df, test_df, feature_columns, y_col = _train_test_frames_for_target(target, train_ids, test_ids)

    rows = []
    for params in param_configs:
        model = _build_model_for_target(target, params)

        X_train, y_train = train_df[feature_columns], train_df[y_col]
        X_test, y_test = test_df[feature_columns], test_df[y_col]
        model.fit(X_train, y_train)

        score_next_value = _score_next_value(model, X_test, y_test, target)
        score_sim_total_cost = _score_sim_total_cost_for_target(target, params, test_ids)

        rows.append(
            {
                "target": target,
                "params": json.dumps(params, sort_keys=True),
                "score_next_value": score_next_value,
                "score_sim_total_cost": score_sim_total_cost,
            }
        )

    return pd.DataFrame(rows, columns=["target", "params", "score_next_value", "score_sim_total_cost"])


def tune_all(targets: list[str], grid: dict, *, partitions: dict | None = None) -> pd.DataFrame:
    frames = []
    for target in targets:
        print(f"\n=== Tuning {target} ===")
        frames.append(tune_single_target(target, grid, partitions=partitions))

    results = pd.concat(frames, ignore_index=True)
    CSV_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(CSV_REPORT_PATH, index=False)
    print(f"Saved tuning CSV to {CSV_REPORT_PATH}")
    return results


if __name__ == '__main__':
    tune_all(MODEL_TARGETS, HYPAM_GRID)