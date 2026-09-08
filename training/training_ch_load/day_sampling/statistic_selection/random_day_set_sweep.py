"""
Statistical experiment: draw many independent random day sets from a fixed pool
of pre-featured CH household-days and score each one, so we can later analyze
what makes a good set / a good day (e.g. which days show up often in good sets).

Procedure:
- fixed pool: the first N train parquet files from random_feature_samples_rsme/train
  (already featured, no feature computation here, just row lookups)
- draw n_candidates independent random day sets of n_days days each from that pool
- for each candidate set:
    - train an XGBoost model (same fixed params as the other net-cost scripts)
    - measure raw prediction RMSE against a fixed CH reference test sample
    - measure raw prediction RMSE against the 20_loads household features
    - simulate net cost on the 20_loads household set (default_scenario, everything
      but base_load predicted by the oracle)
- write candidate_set (all day ids, for reproducibility) | rmse | rmse_20_loads |
  net_cost, one row per candidate, as they are scored
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

# make project modules importable
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
import sys  # noqa: E402
sys.path.insert(0, str(repo_root))

from src.runtime_config import RuntimeConfig  # noqa: E402
from src.simulation.controllers.mpc.predictors.ml.ml_predictor import MLPredictor  # noqa: E402
from src.simulation.controllers.mpc.predictors.ml.model_config import MODEL_FEATURES_BY_FAMILY  # noqa: E402
from src.simulation.controllers.mpc.predictors.modular_predictor import ModularPredictor  # noqa: E402
from src.simulation.controllers.mpc.predictors.oracle.oracle_predictor import OraclePredictor  # noqa: E402
from src.simulation.run_context import RunContext  # noqa: E402
from src.simulation.scenarios.scenario import scenarios as scenario_catalog  # noqa: E402
from src.simulation.simulation import Simulation  # noqa: E402
from src.sqlite_connection import sqlite_conn  # noqa: E402
from training.features.base_load_features import get_base_load_features  # noqa: E402


FEATURE_COLUMNS = MODEL_FEATURES_BY_FAMILY["xgboost"]["base_load"]
TARGET_COLUMN = "next_value"
TARGET = "base_load"
MODEL_PARAMS = {"learning_rate": 0.02, "max_depth": 4, "n_estimators": 100}

RANDOM_FEATURE_DIR = Path(__file__).parents[1] / "random_feature_samples_rsme"
TRAIN_DIR = RANDOM_FEATURE_DIR / "train"
TEST_DIR = RANDOM_FEATURE_DIR / "test"
OUTPUT_DIR = Path(__file__).parent / "random_day_set_sweep_results"


def _load_pool(n_files: int) -> tuple[pd.DataFrame, list[str]]:
    """Concatenate the first n_files train parquet files into one fixed day pool."""
    frames = []
    for seed in range(n_files):
        path = TRAIN_DIR / f"train_seed_{seed}_random_features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing pool file: {path}")
        frames.append(pd.read_parquet(path))

    pool_df = pd.concat(frames, ignore_index=True)
    pool_df["date"] = pd.to_datetime(pool_df["timestamp_utc"]).dt.date.astype(str)
    pool_df["day_key"] = pool_df["household_id"].astype(int).astype(str) + "_" + pool_df["date"]

    day_keys = sorted(pool_df["day_key"].unique().tolist())
    return pool_df, day_keys


def _load_reference_df(test_seed: int) -> pd.DataFrame:
    path = TEST_DIR / f"test_seed_{test_seed}_random_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing reference test sample: {path}")
    return pd.read_parquet(path)


def _score_candidate(
    pool_df: pd.DataFrame,
    day_keys: Sequence[str],
    reference_df: pd.DataFrame,
    reference_df_20_loads: pd.DataFrame,
    test_household_ids: Sequence[int],
    scenario_name: str,
) -> tuple[float, float, float]:
    """Train on the given day keys, return (rmse, rmse_20_loads, net_cost)."""
    train_df = pool_df[pool_df["day_key"].isin(set(day_keys))]

    train_X = train_df[FEATURE_COLUMNS]
    train_y = train_df[TARGET_COLUMN]

    model = XGBRegressor(
        n_estimators=MODEL_PARAMS["n_estimators"],
        max_depth=MODEL_PARAMS["max_depth"],
        learning_rate=MODEL_PARAMS["learning_rate"],
        random_state=42,
        objective="reg:squarederror",
        n_jobs=1,
    )
    model.fit(train_X, train_y)

    rmse = float(root_mean_squared_error(reference_df[TARGET_COLUMN], model.predict(reference_df[FEATURE_COLUMNS])))
    rmse_20_loads = float(
        root_mean_squared_error(
            reference_df_20_loads[TARGET_COLUMN], model.predict(reference_df_20_loads[FEATURE_COLUMNS])
        )
    )

    predictor = ModularPredictor(
        default_predictor=OraclePredictor(),
        target_predictors={
            TARGET: MLPredictor(
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
        scenario=scenario_catalog[scenario_name],
        start_time=1,
    )
    results = simulation.run_batch(
        run_contexts=[run_context],
        household_ids=list(test_household_ids),
        parallel_households=True,
        parallel_workers=6,
        write_results_to_sqlite=False,
    )
    net_cost = float(np.mean(results["net_costs"]))

    return rmse, rmse_20_loads, net_cost


def _print_progress(done: int, total: int, started_at: float) -> None:
    if done == 0:
        print(f"[sweep] [0/{total}] starting...")
        return
    elapsed_seconds = time.perf_counter() - started_at
    avg_seconds = elapsed_seconds / done
    eta_seconds = avg_seconds * max(total - done, 0)
    print(
        f"[sweep] [{done}/{total}] elapsed={elapsed_seconds/60.0:.1f}m "
        f"avg={avg_seconds:.1f}s/run eta={eta_seconds/60.0:.1f}m"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-candidates", type=int, default=200, help="Number of random day sets to draw.")
    parser.add_argument("--n-days", type=int, default=80, help="How many days each candidate set contains.")
    parser.add_argument("--pool-files", type=int, default=5, help="Number of train_seed_*.parquet files forming the fixed pool.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for drawing candidate sets.")
    parser.add_argument("--test-seed", type=int, default=0, help="Reference test sample used for the RMSE score.")
    parser.add_argument("--scenario", type=str, default="default_scenario", help="Simulation scenario used for the net-cost score.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Folder for the output CSV.")
    args = parser.parse_args()

    if args.n_candidates <= 0:
        raise ValueError("--n-candidates must be positive")
    if args.n_days <= 0:
        raise ValueError("--n-days must be positive")
    if args.pool_files <= 0:
        raise ValueError("--pool-files must be positive")

    print(f"Loading fixed day pool from the first {args.pool_files} train parquet file(s)...")
    pool_df, day_keys = _load_pool(args.pool_files)
    print(f"Pool size: {len(day_keys)} unique household-days")
    if len(day_keys) < args.n_days:
        raise ValueError(f"Pool only has {len(day_keys)} unique days, but --n-days requires {args.n_days}.")

    reference_df = _load_reference_df(args.test_seed)
    test_household_ids = list(RuntimeConfig.INDEPENDENT_TEST_SET_20)
    reference_df_20_loads = get_base_load_features(test_household_ids)
    print(f"Scoring net cost on {len(test_household_ids)} households (20_loads), scenario={args.scenario}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"random_day_set_sweep_ncand_{args.n_candidates}_ndays_{args.n_days}_"
        f"poolfiles_{args.pool_files}_seed_{args.seed}"
    )
    output_path = args.output_dir / f"{stem}.csv"
    pd.DataFrame(columns=["candidate_set", "rmse", "rmse_20_loads", "net_cost"]).to_csv(output_path, index=False)  # init early

    rng = np.random.default_rng(args.seed)
    started_at = time.perf_counter()
    _print_progress(0, args.n_candidates, started_at)

    for i in range(1, args.n_candidates + 1):
        candidate_day_keys = sorted(rng.choice(day_keys, size=args.n_days, replace=False).tolist())
        rmse, rmse_20_loads, net_cost = _score_candidate(
            pool_df=pool_df,
            day_keys=candidate_day_keys,
            reference_df=reference_df,
            reference_df_20_loads=reference_df_20_loads,
            test_household_ids=test_household_ids,
            scenario_name=args.scenario,
        )
        print(f"  [{i:4d}/{args.n_candidates}] rmse={rmse:.5f} rmse_20_loads={rmse_20_loads:.5f} net_cost={net_cost:.5f}")
        _print_progress(i, args.n_candidates, started_at)

        row = {
            "candidate_set": ";".join(candidate_day_keys),
            "rmse": round(rmse, 5),
            "rmse_20_loads": round(rmse_20_loads, 5),
            "net_cost": round(net_cost, 5),
        }
        pd.DataFrame([row]).to_csv(output_path, mode="a", header=False, index=False)

    print(f"\nSaved sweep results to {output_path}")


if __name__ == "__main__":
    main()
