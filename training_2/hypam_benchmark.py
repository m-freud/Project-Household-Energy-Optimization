'''
main purpose is to export a csv like this:
(one per model family / target)
eg benchmarks/ridge/base_load/benchmark_results.csv
benchmarks/ridge/base_load/benchmark_summary.csv

columns:
params, next_val_score, sim_score_default_scenario, sim_score_scenario_2, ....

this lets us compare the next val score to net costs of different scenarios,
-> see how prediction accuracy timing impacts different scenarios

'''
from src.sqlite_connection import sqlite_conn

import argparse
import itertools
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import root_mean_squared_error
from simulation.controllers.mpc.predictors.ml.ml_predictor import MLPredictor
from simulation.simulation import Simulation
from src.simulation.controllers.mpc.predictors.modular_predictor import ModularPredictor
import json
import pandas as pd
import numpy as np
from pathlib import Path
from simulation.controllers.mpc.predictors.oracle.oracle_predictor import OraclePredictor
from training.split.clean_split import PARTITIONS
from training._features.base_load_features import get_base_load_features
from training._features.ev_status_features import get_ev_status_features
from training._features.pv_gen_features import get_pv_gen_features
from src.simulation.controllers.mpc.predictors.ml.model_config import MODEL_FEATURES_BY_FAMILY, MODEL_TARGETS


GRID_MAP = {
    "ridge": {
        "grid_1": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
    },
}

BENCHMARK_CSV_DIR = Path(__file__).parent / "benchmarks"

FEATURE_COUNTS = {
    "base_load": len(MODEL_FEATURES_BY_FAMILY["ridge"]["base_load"]),
    "pv_gen": len(MODEL_FEATURES_BY_FAMILY["ridge"]["pv_gen"]),
    "ev_status": len(MODEL_FEATURES_BY_FAMILY["ridge"]["ev_status"]),
}

class HypamBenchmark:
    def __init__(self, model_family, target, grid_id, scenarios, n_test_ids):
        self.model_family = model_family
        self.target = target
        self.scenarios = scenarios
        self.grid = GRID_MAP[model_family][grid_id]
        self.param_configs = self._build_param_grid()
        self.n_test_ids = n_test_ids
        
    def _build_param_grid(self) -> list[dict]:
        keys = list(self.grid.keys())
        return [dict(zip(keys, combo)) for combo in itertools.product(*self.grid.values())]

    def _build_model(self, target: str, params: dict):
        if self.model_family == "ridge":
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

    def _get_train_test_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
        train_ids = PARTITIONS["inner"][self.target]["train"]
        test_ids = PARTITIONS["inner"][self.target]["test"][:self.n_test_ids]
        
        if self.target == "base_load":
            train_df = get_base_load_features(train_ids)
            test_df = get_base_load_features(test_ids)
            y_col = "next_value"
        elif self.target == "pv_gen":
            train_df = get_pv_gen_features(train_ids)
            test_df = get_pv_gen_features(test_ids)
            y_col = "next_value"
        elif self.target in ("ev1_status", "ev2_status"):
            train_df = get_ev_status_features(train_ids)
            test_df = get_ev_status_features(test_ids)
            y_col = "next_state"
        else:
            raise ValueError(f"Unknown target: {self.target}")

        if "ev" in self.target:
            feature_columns = MODEL_FEATURES_BY_FAMILY[self.model_family]["ev_status"]
        else:
            feature_columns = MODEL_FEATURES_BY_FAMILY[self.model_family][self.target]
        return train_df, test_df, feature_columns, y_col

    def benchmark_direct_score(self, model, X_test, y_test):
        """Return the direct model score for the next-value target."""
        if self.target in ("base_load", "pv_gen"):
            return float(root_mean_squared_error(y_test, model.predict(X_test)))

        accuracy = float(model.score(X_test, y_test))
        return float(1.0 - accuracy)
    
    def benchmark_sim_net_cost(self, model, scenario):    
        default_predictor = OraclePredictor()

        target_predictor = MLPredictor(
            base_load_model=model if self.target == "base_load" else None,
            pv_gen_model=model if self.target == "pv_gen" else None,
            ev1_status_model=model if self.target == "ev1_status" else None,
            ev2_status_model=model if self.target == "ev2_status" else None,
        )

        benchmark_predictor = ModularPredictor(
            default_predictor=default_predictor,
            target_predictors={
                self.target: target_predictor,
            },
        )

        sim = Simulation(sqlite_conn, ensure_schema=False)

        return 9.58


    def run_benchmark(self):
        rows = []
        
        train_df, test_df, feature_columns, y_col = self._get_train_test_frames()

        X_train = train_df[feature_columns].to_numpy() # avoid column names, we rely on fixed feature order
        y_train = train_df[y_col].to_numpy()
        X_test = test_df[feature_columns].to_numpy()
        y_test = test_df[y_col].to_numpy()

        for params in self.param_configs:
            model = self._build_model(self.target, params)
            model.fit(X_train, y_train)

            row = {
                "params": json.dumps(params, sort_keys=True),
                "next_val_score": self.benchmark_direct_score(model, X_test, y_test),
            }
            for scenario in self.scenarios:
                row[f"sim_score_{scenario}"] = self.benchmark_sim_net_cost(model, scenario)

            rows.append(row)

        out_path = BENCHMARK_CSV_DIR / self.model_family / f"{self.target}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Saved benchmark to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter benchmarking for selected models and targets.")
    parser.add_argument(
        "--models",
        default="ridge",
        help="Comma-separated model names (default: ridge). Options: ridge, xgb, rf.",
    )
    parser.add_argument(
        "--targets",
        default="base_load",
        help="Comma-separated target names (default: base_load)",
    )
    parser.add_argument(
        "--scenarios",
        default="default_scenario",
        help="Comma-separated scenario names (default: default_scenario)",
    )
    parser.add_argument(
        "--n_test_ids",
        type=int,
        default=18,
        help="Number of test households to use for benchmarking (default: 18)",
    )
    parser.add_argument(
        "--grids",
        default="grid_1",
        help="Comma-separated grid names (default: grid_1)",
    )
    args = parser.parse_args()

    models = args.models.split(",")
    targets = args.targets.split(",")
    scenarios = args.scenarios.split(",")
    n_test_ids = args.n_test_ids
    grids = args.grids.split(",")

    for model_family in models:
        for i, target in enumerate(targets):
            grid_id = grids[i % len(grids)] # Cycle through grids if fewer grids than targets
            benchmark = HypamBenchmark(model_family, target, grid_id, scenarios, n_test_ids)

            benchmark.run_benchmark() # run benchmark, create csv, one per model family / target
