"""Tune CH-trained base-load models by simulated Portugal household net cost."""
from pathlib import Path
import argparse
import json
import sys
import time

# Find the repository root that contains 'src'.
repo_root = next((path for path in Path.cwd().resolve().parents if (path / "src").exists()), "")
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd

from src.simulation.controllers.mpc.predictors.ml.ml_predictor import MLPredictor
from src.simulation.controllers.mpc.predictors.modular_predictor import ModularPredictor
from src.simulation.controllers.mpc.predictors.oracle.oracle_predictor import OraclePredictor
from src.simulation.run_context import RunContext
from src.simulation.scenarios.scenario import scenarios as scenario_catalog
from src.simulation.simulation import Simulation
from src.sqlite_connection import sqlite_conn
from training.tuning.pred_hypam_sweep import (
	GRID_MAP,
	build_model,
	build_param_grid,
	normalize_model,
)
from training.training_ch_load.tuning.pred_hypam_sweep import load_feature_sample_df


OUTPUT_DIR = Path(__file__).parent
TARGET = "base_load"


def score_model(model, scenario_name: str, n_test_ids: int | None) -> float:
	"""Simulate the inner base-load test households with only base-load predicted."""
	from training.split.clean_split import PARTITIONS

	test_ids = PARTITIONS["inner"][TARGET]["test"]
	if n_test_ids is not None:
		test_ids = test_ids[:n_test_ids]

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
		controller_factory=simulation.make_mpc_controller(
			"mpc_iso_benchmark", 96, predictor=predictor
		),
		controller_name="mpc_iso_benchmark",
		scenario=scenario_catalog[scenario_name],
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


def print_progress(done: int, total: int, started_at: float) -> None:
	if done == 0:
		print(f"[sweep] [0/{total}] starting...")
		return
	elapsed_seconds = time.perf_counter() - started_at
	eta_seconds = elapsed_seconds / done * (total - done)
	print(f"[sweep] [{done}/{total}] elapsed={elapsed_seconds / 60:.1f}m eta={eta_seconds / 60:.1f}m")


def hypam_sweep(
	model_family: str,
	grid: str,
	n_days: int,
	seed: int,
	scenarios: list[str],
	n_test_ids: int | None,
) -> None:
	from src.simulation.controllers.mpc.predictors.ml.model_config import MODEL_FEATURES_BY_FAMILY

	features = MODEL_FEATURES_BY_FAMILY[model_family][TARGET]
	train_df = load_feature_sample_df(
		seed=seed,
		n_days=n_days,
		model_family=model_family,
		split="train",
	)
	X_train = train_df[features].to_numpy()
	y_train = train_df["next_value"].to_numpy()
	param_configs = build_param_grid(GRID_MAP[model_family][grid])

	out_dir = OUTPUT_DIR / "net_cost" / model_family
	out_dir.mkdir(parents=True, exist_ok=True)
	output_stem = f"{grid}_seed_{seed}"
	csv_path = out_dir / f"{output_stem}.csv"
	json_path = out_dir / f"{output_stem}.json"
	pd.DataFrame(columns=["params", "scenario", "score"]).to_csv(csv_path, index=False)

	total_runs = len(param_configs) * len(scenarios)
	completed_runs = 0
	started_at = time.perf_counter()
	print(
		f"Running CH net-cost sweep: model={model_family}, grid={grid}, "
		f"train_seed={seed}, train_days={n_days}, scenarios={scenarios}"
	)
	print_progress(completed_runs, total_runs, started_at)
	for index, params in enumerate(param_configs, 1):
		estimator = build_model(model_family, TARGET, params)
		estimator.fit(X_train, y_train)
		for scenario_name in scenarios:
			score = score_model(estimator, scenario_name, n_test_ids)
			completed_runs += 1
			print(
				f"  [{index:3d}/{len(param_configs)}] params={params} "
				f"scenario={scenario_name} -> net_cost={score:.5f}"
			)
			print_progress(completed_runs, total_runs, started_at)
			pd.DataFrame(
				[{"params": json.dumps(params, sort_keys=True), "scenario": scenario_name, "score": round(score, 5)}]
			).to_csv(csv_path, mode="a", header=False, index=False)

	with open(json_path, "w") as file:
		json.dump(
			{
				"grid": GRID_MAP[model_family][grid],
				"target": TARGET,
				"model": model_family,
				"ch_train_split": "train",
				"ch_train_seed": seed,
				"ch_training_days": n_days,
				"scenarios": scenarios,
			},
			file,
			indent=2,
		)
	print(f"Saved sweep results to {csv_path}")
	print(f"Saved sweep metadata to {json_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", nargs="+", default=["xgboost"])
	parser.add_argument("--grid", nargs="+", default=["load_2"])
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--n_days", type=int, default=120)
	parser.add_argument("--scenario", nargs="+", default=["default_scenario", "00_baseline"])
	parser.add_argument("--n_test_ids", type=int, default=None)
	args = parser.parse_args()

	for raw_model in args.model:
		model_family = normalize_model(raw_model)
		for grid_name in args.grid:
			if grid_name not in GRID_MAP.get(model_family, {}):
				print(f"Skipping {model_family}/{grid_name}: grid not defined for model")
				continue
			hypam_sweep(
				model_family=model_family,
				grid=grid_name,
				n_days=args.n_days,
				seed=args.seed,
				scenarios=args.scenario,
				n_test_ids=args.n_test_ids,
			)
