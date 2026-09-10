"""Compare Portugal-global and CH-selected training data over the load_2 grid."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = next((path for path in Path.cwd().resolve().parents if (path / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.controllers.mpc.predictors.ml.model_config import MODEL_FEATURES_BY_FAMILY  # noqa: E402
from simulation.controllers.mpc.predictors.ml.recursive.recursive_ml_predictor import RecursiveMLPredictor  # noqa: E402
from src.simulation.controllers.mpc.predictors.modular_predictor import ModularPredictor  # noqa: E402
from src.simulation.controllers.mpc.predictors.oracle.oracle_predictor import OraclePredictor  # noqa: E402
from src.simulation.run_context import RunContext  # noqa: E402
from src.simulation.scenarios.scenario import scenarios as scenario_catalog  # noqa: E402
from src.simulation.simulation import Simulation  # noqa: E402
from src.sqlite_connection import sqlite_conn  # noqa: E402
from training.split.clean_split import PARTITIONS  # noqa: E402
from training.tuning.pred_hypam_sweep import GRID_MAP, build_model, build_param_grid  # noqa: E402
from training.training_ch_load.day_sampling.genetic_selection import (  # noqa: E402
    FEATURE_TABLE_NAME,
)
from training.training_ch_load.day_sampling.sampling import SQLITE_PATH  # noqa: E402
from training.features.base_load_features import get_base_load_features  # noqa: E402


TOP_SETS_PATH = (
    Path(__file__).parent
    / "training_ch_load"
    / "day_sampling"
    / "genetic_selection_results"
    / "rmse_evolution_n_days_150_pop_150_gen_42_mut_0.2_seed_42_top_sets.csv"
)
OUTPUT_PATH = Path(__file__).parent / "ch_vs_portugal_load_bakeoff_load_2.csv"
MODEL_FAMILY = "xgboost"
TARGET = "base_load"


def _load_best_ch_day_set(path: Path) -> list[tuple[int, str]]:
    top_sets = pd.read_csv(path)
    if top_sets.empty or "day_set" not in top_sets.columns:
        raise ValueError(f"Top-set CSV has no usable day_set column: {path}")

    serialized_days = str(top_sets.iloc[0]["day_set"])
    day_set = []
    for item in serialized_days.split(";"):
        household_id, date = item.rsplit("_", 1)
        day_set.append((int(household_id), date))
    return day_set


def _load_ch_features(day_set: list[tuple[int, str]]) -> pd.DataFrame:
    placeholders = ", ".join("(?, ?)" for _ in day_set)
    params = tuple(value for pair in day_set for value in pair)
    query = (
        f"SELECT * FROM {FEATURE_TABLE_NAME} "
        f"WHERE (household_id, date) IN ({placeholders})"
    )
    with sqlite3.connect(SQLITE_PATH) as connection:
        features = pd.read_sql_query(query, connection, params=params)

    if features.empty:
        raise ValueError("No rows found in load_features for the selected CH day set")
    return features


def _score_default_scenario(model: object, test_ids: list[int]) -> float:
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
        controller_factory=simulation.make_mpc_controller(
            "mpc_iso_benchmark", 96, predictor=predictor
        ),
        controller_name="mpc_iso_benchmark",
        scenario=scenario_catalog["default_scenario"],
        start_time=1,
    )
    results = simulation.run_batch(
        run_contexts=[run_context],
        household_ids=test_ids,
        parallel_households=True,
        parallel_workers=6,
        write_results_to_sqlite=False,
    )
    return float(np.mean(results["net_costs"]))


def _score_training_data(
    train_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    params: dict,
) -> tuple[object, float]:
    features = MODEL_FEATURES_BY_FAMILY[MODEL_FAMILY][TARGET]
    model = build_model(MODEL_FAMILY, TARGET, params)
    model.fit(train_df[features], train_df["next_value"])
    prediction_rmse = float(
        ((reference_df["next_value"] - model.predict(reference_df[features])) ** 2).mean() ** 0.5
    )
    return model, prediction_rmse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--n-test-ids", type=int, default=None)
    parser.add_argument("--max-configs", type=int, default=None)
    args = parser.parse_args()

    features = MODEL_FEATURES_BY_FAMILY[MODEL_FAMILY][TARGET]
    global_train_ids = sorted(PARTITIONS["global"]["train"])
    global_test_ids = list(PARTITIONS["global"]["test"])
    if args.n_test_ids is not None:
        global_test_ids = global_test_ids[: args.n_test_ids]
    portugal_train = get_base_load_features(global_train_ids)
    reference_df = get_base_load_features(global_test_ids)

    ch_day_set = _load_best_ch_day_set(TOP_SETS_PATH)
    ch_train = _load_ch_features(ch_day_set)
    ch_train["next_value"] = pd.to_numeric(ch_train["next_value"], errors="coerce")
    for column in features:
        ch_train[column] = pd.to_numeric(ch_train[column], errors="coerce")

    param_configs = build_param_grid(GRID_MAP[MODEL_FAMILY]["load_2"])
    if args.max_configs is not None:
        if args.max_configs <= 0:
            raise ValueError("--max-configs must be positive")
        param_configs = param_configs[: args.max_configs]
    started_at = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["data_source", "params", "prediction_rmse", "default_scenario_net_cost"]
    pd.DataFrame(columns=columns).to_csv(args.output, index=False)

    print(f"Portugal global train IDs: {len(global_train_ids)}")
    print(f"Portugal evaluation IDs: {len(global_test_ids)}")
    print(f"CH selected day pairs: {len(ch_day_set)} ({len(ch_train)} feature rows)")
    print(f"Grid configurations: {len(param_configs)}; scenario: default_scenario")

    for index, params in enumerate(param_configs, start=1):
        print(f"[{index}/{len(param_configs)}] params={params}")
        portugal_model, portugal_rmse = _score_training_data(portugal_train, reference_df, params)
        ch_model, ch_rmse = _score_training_data(ch_train, reference_df, params)
        portugal_cost = _score_default_scenario(portugal_model, global_test_ids)
        ch_cost = _score_default_scenario(ch_model, global_test_ids)
        pd.DataFrame(
            [
                {"data_source": "portugal_global", "params": json.dumps(params, sort_keys=True), "prediction_rmse": portugal_rmse, "default_scenario_net_cost": portugal_cost},
                {"data_source": "ch_best_set", "params": json.dumps(params, sort_keys=True), "prediction_rmse": ch_rmse, "default_scenario_net_cost": ch_cost},
            ],
            columns=columns,
        ).to_csv(args.output, mode="a", header=False, index=False)
        elapsed = time.perf_counter() - started_at
        remaining = elapsed / index * (len(param_configs) - index)
        print(f"  Portugal: RMSE={portugal_rmse:.6f}, net_cost={portugal_cost:.6f}")
        print(f"  CH:       RMSE={ch_rmse:.6f}, net_cost={ch_cost:.6f}")
        print(
            f"  progress={index}/{len(param_configs)} "
            f"elapsed={elapsed / 60:.1f} min "
            f"eta={remaining / 60:.1f} min"
        )

    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()