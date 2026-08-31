'''
same as pred_hypam_sweep, but scores each candidate model by running full simulations
and measuring net cost, instead of scoring against the held-out val set directly.

every other target is fixed to the oracle predictor, so only the swept target's
predictor varies -> isolates the effect of that predictor on net cost.
'''
# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

import argparse  # noqa
import json  # noqa
import time  # noqa
import numpy as np  # noqa
import pandas as pd  # noqa

from src.simulation.controllers.mpc.predictors.ml.ml_predictor import MLPredictor  # noqa
from src.simulation.controllers.mpc.predictors.modular_predictor import ModularPredictor  # noqa
from src.simulation.controllers.mpc.predictors.oracle.oracle_predictor import OraclePredictor  # noqa
from src.simulation.run_context import RunContext  # noqa
from src.simulation.scenarios.scenario import scenarios as scenario_catalog  # noqa
from src.simulation.simulation import Simulation  # noqa
from src.sqlite_connection import sqlite_conn  # noqa
from training.split.clean_split import PARTITIONS  # noqa

from pred_hypam_sweep import (  # noqa
    GRID_MAP,
    OUTPUT_DIR,
    build_model,
    build_param_grid,
    get_train_test_frames,
    normalize_model,
    normalize_target,
)

def score_model(model, target: str, scenario_name: str, n_test_ids: int | None) -> float:
    '''run a batch simulation with `target` predicted by `model` and everything else oracle, return avg net cost'''
    test_ids = PARTITIONS["inner"][target]["test"]
    if n_test_ids:
        test_ids = test_ids[:n_test_ids]

    scenario = scenario_catalog[scenario_name]

    target_predictor = MLPredictor(
        base_load_model=model if target == "base_load" else None,
        pv_gen_model=model if target == "pv_gen" else None,
        ev1_status_model=model if target == "ev1_status" else None,
        ev2_status_model=model if target == "ev2_status" else None,
    )

    predictor = ModularPredictor(
        default_predictor=OraclePredictor(),
        target_predictors={target: target_predictor},
    )

    sim = Simulation(sqlite_conn, ensure_results_table=False)

    run_context = RunContext(
        controller_factory=sim.make_mpc_controller("mpc_iso_benchmark", 96, predictor=predictor),
        controller_name="mpc_iso_benchmark",
        scenario=scenario,
        start_time=1,
    )

    results = sim.run_batch(
        run_contexts=[run_context],
        household_ids=test_ids,
        parallel_households=True,
        parallel_workers=6,
        write_results_to_sqlite=False,
    )

    return float(np.mean(results["net_costs"]))


def _print_progress(done: int, total: int, started_at: float, label: str = "sweep"):
    if done == 0:
        print(f"[{label}] [0/{total}] starting...")
        return

    elapsed_seconds = time.perf_counter() - started_at
    avg_seconds = elapsed_seconds / done
    eta_seconds = avg_seconds * max(total - done, 0)
    print(
        f"[{label}] [{done}/{total}] elapsed={elapsed_seconds/60.0:.1f}m "
        f"avg={avg_seconds:.1f}s/run eta={eta_seconds/60.0:.1f}m"
    )


def hypam_sweep(target: str, model: str, grid: str, scenarios: list[str], n_test_ids: int | None, full_progress: dict | None = None):
    '''do a hypam sweep by training models for all grid configs and scoring them via simulated net cost'''

    train_df, _test_df, feature_columns, y_col = get_train_test_frames(target=target, model_family=model)

    X_train = train_df[feature_columns].to_numpy()  # avoid column names, we rely on fixed feature order
    y_train = train_df[y_col].to_numpy()

    out_dir = OUTPUT_DIR / "net_cost" / target / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{grid}.csv"
    json_path = out_dir / f"{grid}.json"

    print(
        f"Running net-cost hyperparameter sweep for target: {target}, model: {model}, "
        f"grid: {grid}, scenarios: {scenarios}"
    )

    output_columns = ["params", "scenario", "score"]
    pd.DataFrame(columns=output_columns).to_csv(out_path, index=False)  # init file early so partial progress survives crashes

    param_configs = build_param_grid(GRID_MAP[model][grid])
    total_runs = len(param_configs) * len(scenarios)
    done_runs = 0
    started_at = time.perf_counter()
    _print_progress(done_runs, total_runs, started_at, label="current sweep")

    for i, params in enumerate(param_configs, 1):
        estimator = build_model(model, target, params)
        estimator.fit(X_train, y_train)

        for scenario in scenarios:
            score = score_model(estimator, target, scenario, n_test_ids)
            done_runs += 1
            print(f"  [{i:3d}/{len(param_configs)}] params={params} scenario={scenario} -> net_cost={score:.5f}")
            _print_progress(done_runs, total_runs, started_at, label="current sweep")
            if full_progress is not None:
                full_progress["done"] += 1
                _print_progress(
                    full_progress["done"],
                    full_progress["total"],
                    full_progress["started_at"],
                    label="full run",
                )
            row = {
                "params": json.dumps(params, sort_keys=True),
                "scenario": scenario,
                "score": round(score, 5),
            }
            pd.DataFrame([row]).to_csv(out_path, mode="a", header=False, index=False)

    with open(json_path, "w") as f:
        json.dump(GRID_MAP[model][grid], f, indent=2)
    print(f"Saved sweep results to {out_path}")
    print(f"Saved hypam grid to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        nargs="+",
        default=["pv_gen"],
        help="target(s). choose between base_load,pv_gen,ev1_status,ev2_status",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["xgboost"],
        help="inference model(s). choose between xgboost, random_forest, ridge",
    )
    parser.add_argument(
        "--grid",
        type=str,
        nargs="+",
        default=["grid_1"],
        help="grid(s)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        nargs="+",
        default=["default_scenario", "00_baseline"],
        help="scenario(s) used for the simulation runs",
    )
    parser.add_argument(
        "--n_test_ids",
        type=int,
        default=None,
        help="number of test households used per simulation (default: all)",
    )
    args = parser.parse_args()

    targets = [normalize_target(t) for t in args.target]
    if targets == ["all"]:
        targets = ["base_load", "pv_gen", "ev1_status", "ev2_status"]
    models = [normalize_model(m) for m in args.model]
    grids = args.grid

    full_total_runs = 0
    for model in models:
        for target in targets:
            for grid in grids:
                if grid in GRID_MAP.get(model, {}):
                    full_total_runs += len(build_param_grid(GRID_MAP[model][grid])) * len(args.scenario)

    full_progress = {
        "done": 0,
        "total": full_total_runs,
        "started_at": time.perf_counter(),
    }
    _print_progress(0, full_total_runs, full_progress["started_at"], label="full run")

    for model in models:
        for target in targets:
            for grid in grids:
                if grid not in GRID_MAP.get(model, {}):
                    print(f"Skipping {target}/{model}/{grid}: grid not defined for model")
                    continue
                hypam_sweep(
                    target=target,
                    model=model,
                    grid=grid,
                    scenarios=args.scenario,
                    n_test_ids=args.n_test_ids,
                    full_progress=full_progress,
                )