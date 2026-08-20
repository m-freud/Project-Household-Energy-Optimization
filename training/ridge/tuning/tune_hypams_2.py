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
from functools import partial
from pathlib import Path
from statistics import fmean

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.runtime_config import RuntimeConfig
from src.simulation.controllers.mpc.predictors.ml.helpers.base_load import predict_base_load
from src.simulation.controllers.mpc.predictors.ml.helpers.ev_status import _predict_single_ev_status
from src.simulation.controllers.mpc.predictors.ml.helpers.pv_gen import predict_pv_gen
from src.simulation.controllers.mpc.predictors.ml.model_config import MODEL_FEATURES_BY_FAMILY, MODEL_TARGETS
from src.simulation.controllers.mpc.predictors.modular_predictor import ModularPredictor
from src.simulation.controllers.mpc.predictors.oracle.oracle_predictor import OraclePredictor
from src.simulation.controllers.mpc.predictors.ml.ml_predictor import MLPredictor
from src.simulation.run_context import RunContext
from src.simulation.scenarios.scenario import scenarios as scenario_catalog
from src.simulation.simulation import Simulation, build_mpc_controller
from src.sqlite_connection import sqlite_conn
from training._features.base_load_features import get_base_load_features
from training._features.ev_status_features import get_ev_status_features
from training._features.pv_gen_features import get_pv_gen_features
from training.split.clean_split import PARTITIONS

CSV_REPORT_PATH = Path(RuntimeConfig.ROOT_DIR) / "training" / "ridge" / "tuning" / "tuning_results_18.csv"

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


def _fit_lightweight_placeholder_model(target: str, feature_count: int):
    """Return a fitted dummy model for inactive slots in the target-specific modular scorer.

    These placeholders are never used in the active target path, but they must be
    fitted so the shared `MLPredictor` object does not raise `NotFittedError` when
    the modular controller checks the other model slots during prediction.
    """
    if target in ("base_load", "pv_gen"):
        model = DummyRegressor(strategy="mean")
        X = np.zeros((1, max(feature_count, 1)), dtype=float)
        y = np.array([0.0], dtype=float)
        model.fit(X, y)
        return model

    model = DummyClassifier(strategy="most_frequent")
    X = np.zeros((1, max(feature_count, 1)), dtype=float)
    y = np.array([0], dtype=int)
    model.fit(X, y)
    return model


def _score_next_value(model, X_test: pd.DataFrame, y_test: pd.Series, target: str) -> float:
    """Return the direct model score for the next-value target."""
    if target in ("base_load", "pv_gen"):
        return float(root_mean_squared_error(y_test, model.predict(X_test)))

    accuracy = float(model.score(X_test, y_test))
    return float(1.0 - accuracy)


def _score_sim_total_cost_for_target(
    model,
    target: str,
    params: dict,
    test_households: list[int],
    scenarios: list[str] | None = None,
) -> float:
    """Use the already-fit model inside a reusable ML predictor, then evaluate
    the target-specific modular MPC against the oracle-default baseline.
    """
    scenario_names = list(scenarios or ["default_scenario"])
    target_key = {
        "base_load": "base_load",
        "pv_gen": "pv_gen",
        "ev1_status": "ev_status",
        "ev2_status": "ev_status",
    }.get(target)
    if target_key is None:
        raise ValueError(f"Unsupported target for simulation score: {target}")

    default_predictor = OraclePredictor()
    dummy_feature_count = {
        "base_load": len(MODEL_FEATURES_BY_FAMILY["ridge"]["base_load"]),
        "pv_gen": len(MODEL_FEATURES_BY_FAMILY["ridge"]["pv_gen"]),
        "ev_status": len(MODEL_FEATURES_BY_FAMILY["ridge"]["ev_status"]),
    }

    dummy_regressor = _fit_lightweight_placeholder_model("base_load", dummy_feature_count["base_load"])
    dummy_classifier = _fit_lightweight_placeholder_model("ev1_status", dummy_feature_count["ev_status"])

    runtime_predictor = MLPredictor(
        base_load_model=(model if target == "base_load" else dummy_regressor),
        pv_gen_model=(model if target == "pv_gen" else dummy_regressor),
        ev1_status_model=(model if target == "ev1_status" else dummy_classifier),
        ev2_status_model=(model if target == "ev2_status" else dummy_classifier),
    )

    sim = Simulation(sqlite_conn, ensure_schema=False)
    all_total_costs: list[float] = []

    for scenario_name in scenario_names:
        scenario = scenario_catalog[scenario_name]
        policy_name = f"tune_{target}_{scenario_name}"
        controller_factory = partial(
            build_mpc_controller,
            name=policy_name,
            horizon=96,
            predictor=ModularPredictor(
                default_predictor=default_predictor,
                target_predictors={target_key: runtime_predictor},
            ),
            duration_hours=sim.duration_hours,
        )

        run_contexts = [
            RunContext(
                controller_factory=controller_factory,
                controller_name=policy_name,
                scenario=scenario,
                start_time=1,
            )
            for _ in test_households
        ]

        results = sim.run_batch(
            run_contexts,
            household_ids=test_households,
            max_households=len(test_households),
            parallel_households=True,
            parallel_workers=6,
            write_results_to_sqlite=False,
        )

        total_costs = results.get("total_costs", [])
        if not total_costs:
            raise RuntimeError(
                f"No total_cost results returned for {scenario_name}/{policy_name}."
            )
        all_total_costs.extend(float(value) for value in total_costs)

    if not all_total_costs:
        return 0.0

    return float(fmean(all_total_costs))


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
    test_ids = partition_map["inner"][target]["test"][:18]

    train_df, test_df, feature_columns, y_col = _train_test_frames_for_target(target, train_ids, test_ids)

    rows = []
    for params in param_configs:
        model = _build_model_for_target(target, params)

        X_train = train_df[feature_columns].to_numpy()
        y_train = train_df[y_col].to_numpy()
        X_test = test_df[feature_columns].to_numpy()
        y_test = test_df[y_col].to_numpy()
        model.fit(X_train, y_train)

        score_next_value = _score_next_value(model, X_test, y_test, target)
        score_sim_total_cost = _score_sim_total_cost_for_target(model, target, params, test_ids)

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
    """Return one CSV-ready row per candidate parameter set for all targets."""
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