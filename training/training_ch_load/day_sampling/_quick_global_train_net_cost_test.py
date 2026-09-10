"""Quick sanity test: train XGBoost on the global train partition and measure net cost on 20_loads."""

from pathlib import Path
import sys

# make project modules importable
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

import numpy as np
from xgboost import XGBRegressor

from src.runtime_config import RuntimeConfig
from src.simulation.controllers.mpc.predictors.ml.recursive.recursive_ml_predictor import RecursiveMLPredictor
from src.simulation.controllers.mpc.predictors.ml.model_config import MODEL_FEATURES_BY_FAMILY
from src.simulation.controllers.mpc.predictors.modular_predictor import ModularPredictor
from src.simulation.controllers.mpc.predictors.oracle.oracle_predictor import OraclePredictor
from src.simulation.run_context import RunContext
from src.simulation.scenarios.scenario import scenarios as scenario_catalog
from src.simulation.simulation import Simulation
from src.sqlite_connection import sqlite_conn
from training.features.base_load_features import get_base_load_features
from training.split.clean_split import PARTITIONS

MODEL_PARAMS = {"learning_rate": 0.02, "max_depth": 4, "n_estimators": 100}
TARGET = "base_load"
FEATURE_COLUMNS = MODEL_FEATURES_BY_FAMILY["xgboost"][TARGET]


def main() -> None:
    train_ids = sorted(PARTITIONS["global"]["train"])
    train_df = get_base_load_features(train_ids)

    model = XGBRegressor(
        n_estimators=MODEL_PARAMS["n_estimators"],
        max_depth=MODEL_PARAMS["max_depth"],
        learning_rate=MODEL_PARAMS["learning_rate"],
        random_state=42,
        objective="reg:squarederror",
        n_jobs=1,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["next_value"])

    predictor = ModularPredictor(
        default_predictor=OraclePredictor(),
        target_predictors={
            TARGET: RecursiveMLPredictor(
                base_load_model=model,
                pv_gen_model=None,
                ev1_status_model=None,
                ev2_status_model=None,
            )
        },
    )
    simulation = Simulation(sqlite_conn, ensure_results_table=False)
    run_context = RunContext(
        controller_factory=simulation.make_mpc_controller("mpc_iso_benchmark", 96, predictor=predictor),
        controller_name="mpc_iso_benchmark",
        scenario=scenario_catalog["default_scenario"],
        start_time=1,
    )
    results = simulation.run_batch(
        run_contexts=[run_context],
        household_ids=list(RuntimeConfig.INDEPENDENT_TEST_SET_20),
        parallel_households=True,
        parallel_workers=6,
        write_results_to_sqlite=False,
    )

    print(f"net_cost (default_scenario, 20_loads) = {float(np.mean(results['net_costs'])):.5f}")


if __name__ == "__main__":
    main()

