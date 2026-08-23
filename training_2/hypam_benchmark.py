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
# paste this to enable src. imports
from pathlib import Path
import sys

from xgboost import XGBRegressor, XGBClassifier

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))
from src.simulation.run_context import RunContext
from src.sqlite_connection import sqlite_conn

import argparse
import itertools
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import root_mean_squared_error
from src.simulation.controllers.mpc.predictors.ml.ml_predictor import MLPredictor
from src.simulation.simulation import Simulation
from src.simulation.controllers.mpc.predictors.modular_predictor import ModularPredictor
import json
import pandas as pd
import numpy as np
from pathlib import Path
from src.simulation.controllers.mpc.predictors.oracle.oracle_predictor import OraclePredictor
from training.split.clean_split import PARTITIONS
from training._features.base_load_features import get_base_load_features
from training._features.ev_status_features import get_ev_status_features
from training._features.pv_gen_features import get_pv_gen_features
from src.simulation.controllers.mpc.predictors.ml.model_config import MODEL_FEATURES_BY_FAMILY, MODEL_TARGETS
from src.simulation.scenarios.scenario import scenarios as scenario_catalog

GRID_MAP = {
    "ridge": {
        "grid_1": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
    },
    "xgboost": {
        "grid_1": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
        },
    },
}

BENCHMARK_CSV_DIR = Path(__file__).parent / "benchmarks"

FEATURE_COUNTS = {
    "base_load": len(MODEL_FEATURES_BY_FAMILY["ridge"]["base_load"]),
    "pv_gen": len(MODEL_FEATURES_BY_FAMILY["ridge"]["pv_gen"]),
    "ev_status": len(MODEL_FEATURES_BY_FAMILY["ridge"]["ev_status"]),
}

class HypamBenchmark: #TODO round floats in csv
    def __init__(self, model_family, target, grid_id, scenario_names, n_test_ids):
        self.model_family = model_family
        self.target = target
        self.scenarios = [scenario_catalog[name] for name in scenario_names]
        self.grid_id = grid_id
        self.grid = GRID_MAP[model_family][grid_id]
        self.param_configs = self._build_param_grid()
        self.n_test_ids = n_test_ids

    def _print_benchmark_sim_progress(self):
        elapsed_seconds = time.perf_counter() - self._progress_started_at
        done = self._progress_done
        total = self._progress_total
        remaining = max(total - done, 0)

        if done == 0:
            print(f"[0/{total}] starting simulation benchmarks...")
            return

        avg_seconds_per_sim = elapsed_seconds / done
        eta_seconds = avg_seconds_per_sim * remaining
        print(
            f"[{done}/{total}] "
            f"elapsed={elapsed_seconds/60.0:.1f}m "
            f"avg={avg_seconds_per_sim:.1f}s/sim "
            f"eta={eta_seconds/60.0:.1f}m"
        )
        
    def _build_param_grid(self) -> list[dict]:
        keys = list(self.grid.keys())
        return [dict(zip(keys, combo)) for combo in itertools.product(*self.grid.values())]

    def _build_model(self, target: str, params: dict):
        if target in ("base_load", "pv_gen"):
            if self.model_family == "ridge":
                estimator = Ridge(**params)
            if self.model_family == "xgboost":
                estimator = XGBRegressor(**params)
        else:
            if self.model_family == "ridge":
                estimator = RidgeClassifier(**params)
            if self.model_family == "xgboost":
                estimator = XGBClassifier(**params)


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

        sim = Simulation(sqlite_conn, ensure_results_table=False)

        run_context = RunContext(
            controller_factory=sim.make_mpc_controller("mpc_iso_benchmark", 96, predictor=benchmark_predictor),
            controller_name="mpc_iso_benchmark",
            scenario=scenario,
            start_time=1,
        )

        results = sim.run_batch(
            run_contexts=[run_context],
            household_ids=PARTITIONS["inner"][self.target]["test"][:self.n_test_ids],
            parallel_households=True,
            parallel_workers=6,
            write_results_to_sqlite=False
        )

        avg_net_cost = np.mean(results["net_costs"])

        return avg_net_cost


    def run_benchmark(self):
        out_path = BENCHMARK_CSV_DIR / self.model_family / f"{self.target}_{self.n_test_ids}_{self.grid_id}_{'_'.join([scenario.name for scenario in self.scenarios])}_paul.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize output file early so partial progress survives crashes.
        output_columns = ["params", "next_val_score", "scenario", "sim_score"]
        pd.DataFrame(columns=output_columns).to_csv(out_path, index=False)

        self._progress_total = len(self.param_configs) * len(self.scenarios)
        self._progress_done = 0
        self._progress_started_at = time.perf_counter()
        self._print_benchmark_sim_progress()
        
        train_df, test_df, feature_columns, y_col = self._get_train_test_frames()

        X_train = train_df[feature_columns].to_numpy() # avoid column names, we rely on fixed feature order
        y_train = train_df[y_col].to_numpy()
        X_test = test_df[feature_columns].to_numpy()
        y_test = test_df[y_col].to_numpy()

        for params in self.param_configs:
            model = self._build_model(self.target, params)
            model.fit(X_train, y_train)

            params_json = json.dumps(params, sort_keys=True)
            next_val_score = self.benchmark_direct_score(model, X_test, y_test)

            for scenario in self.scenarios:
                sim_score = self.benchmark_sim_net_cost(model, scenario)
                pd.DataFrame(
                    [
                        {
                            "params": params_json,
                            "next_val_score": next_val_score,
                            "scenario": scenario.name,
                            "sim_score": sim_score,
                        }
                    ]
                ).to_csv(out_path, mode="a", header=False, index=False)
                self._progress_done += 1
                self._print_benchmark_sim_progress()

        print(f"Saved benchmark to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter benchmarking for selected models and targets.")
    parser.add_argument(
        "--models",
        default="ridge",
        help="Comma-separated model names (default: ridge). Options: ridge, xgboost, rf.",
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
