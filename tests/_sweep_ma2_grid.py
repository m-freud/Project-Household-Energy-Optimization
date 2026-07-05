from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# Keep imports working when script is run directly from terminal.
repo_root = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").exists()), None)
if repo_root is None:
    raise RuntimeError("Could not find repository root containing 'src'")
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.simulation.controllers.mpc.predictors.moving_average_predictor2 import MovingAveragePredictor2
from src.simulation.run_context import RunContext
from src.simulation.scenarios.scenario import scenarios as scenario_catalog
from src.simulation.simulation import Simulation, make_mpc_controller


def parse_int_list(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid sweep for MovingAveragePredictor2")
    parser.add_argument("--households", default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")
    parser.add_argument("--short-windows", default="2,4,8")
    parser.add_argument("--long-windows", default="12,24,48")
    parser.add_argument("--short-weight", type=float, default=0.7)
    parser.add_argument("--parallel-workers", type=int, default=None)
    parser.add_argument("--run-tag", default="ma2_grid")
    args = parser.parse_args()

    households = parse_int_list(args.households)
    short_windows = parse_int_list(args.short_windows)
    long_windows = parse_int_list(args.long_windows)
    short_weight = float(args.short_weight)

    baseline_policies = ["waterfall", "mpc_oracle", "mpc_moving_average", "mpc_moving_average2"]
    baseline_run_ids = {
        "waterfall": "1",
        "mpc_oracle": "1",
        "mpc_moving_average": "2",
        "mpc_moving_average2": "12",
    }

    print("Starting MA2 sweep", flush=True)
    print(f"Households: {households}", flush=True)
    print(f"Short windows: {short_windows}", flush=True)
    print(f"Long windows: {long_windows}", flush=True)
    print(f"Short weight: {short_weight}", flush=True)

    sim_conn = sqlite3.connect(Config.SQLITE_PATH)
    combo_rows: list[tuple[str, int, int, str]] = []

    try:
        sim = Simulation(sim_conn)

        for short_window in short_windows:
            for long_window in long_windows:
                if long_window < short_window:
                    continue

                policy_name = f"mpc_ma2_s{short_window}_l{long_window}_{args.run_tag}"
                print(
                    f"\nRunning policy={policy_name} (short={short_window}, long={long_window})",
                    flush=True,
                )

                controller_factory = make_mpc_controller(
                    policy_name,
                    horizon=96,
                    predictor=MovingAveragePredictor2(
                        short_window_size=short_window,
                        long_window_size=long_window,
                        short_weight=short_weight,
                    ),
                )

                run_contexts = [
                    RunContext(
                        controller_factory=controller_factory,
                        controller_name=policy_name,
                        scenario=scenario,
                        start_time=1,
                    )
                    for scenario in scenario_catalog
                ]

                run_id = run_contexts[0].run_id
                print(f"Assigned run_id={run_id}", flush=True)

                sim.run_batch(
                    run_contexts,
                    household_ids=households,
                    parallel_households=True,
                    parallel_workers=args.parallel_workers,
                )

                combo_rows.append((policy_name, short_window, long_window, run_id))

    finally:
        sim_conn.close()

    print("\nAll sweep runs completed. Building comparison summary...", flush=True)

    analysis_conn = sqlite3.connect(Config.SQLITE_PATH)
    try:
        df = pd.read_sql_query(
            """
            SELECT run_id, policy, scenario, player_id,
                   total_cost, net_cost, net_load,
                   target_met_all_bess, target_met_all_ev1, target_met_all_ev2
            FROM results
            """,
            analysis_conn,
        )
    finally:
        analysis_conn.close()

    rows: list[dict] = []
    for policy_name, short_window, long_window, run_id in combo_rows:
        subset = df[
            (df["player_id"].isin(households))
            & (
                ((df["policy"] == policy_name) & (df["run_id"] == run_id))
                | ((df["policy"].isin(baseline_policies)) & (df["run_id"] == df["policy"].map(baseline_run_ids)))
            )
        ].copy()

        counts = subset.groupby(["scenario", "player_id"])["policy"].nunique().reset_index(name="n")
        valid = counts[counts["n"] == 5][["scenario", "player_id"]]
        subset = subset.merge(valid, on=["scenario", "player_id"], how="inner")

        pivot = subset.pivot_table(
            index=["scenario", "player_id"],
            columns="policy",
            values=[
                "total_cost",
                "net_cost",
                "net_load",
                "target_met_all_bess",
                "target_met_all_ev1",
                "target_met_all_ev2",
            ],
            aggfunc="first",
        )
        pivot.columns = [f"{metric}__{policy}" for metric, policy in pivot.columns]
        comp = pivot.reset_index()

        for metric in ["total_cost", "net_cost", "net_load"]:
            comp[f"{metric}_delta_vs_waterfall"] = comp[f"{metric}__{policy_name}"] - comp[f"{metric}__waterfall"]
            comp[f"{metric}_delta_vs_oracle"] = comp[f"{metric}__{policy_name}"] - comp[f"{metric}__mpc_oracle"]
            comp[f"{metric}_delta_vs_ma1"] = comp[f"{metric}__{policy_name}"] - comp[f"{metric}__mpc_moving_average"]
            comp[f"{metric}_delta_vs_ma2_base"] = comp[f"{metric}__{policy_name}"] - comp[f"{metric}__mpc_moving_average2"]

        rows.append(
            {
                "policy": policy_name,
                "run_id": run_id,
                "short_window": short_window,
                "long_window": long_window,
                "pairs": int(comp.shape[0]),
                "avg_total_cost": float(comp[f"total_cost__{policy_name}"].mean()),
                "avg_total_delta_vs_waterfall": float(comp["total_cost_delta_vs_waterfall"].mean()),
                "avg_total_delta_vs_oracle": float(comp["total_cost_delta_vs_oracle"].mean()),
                "avg_total_delta_vs_ma1": float(comp["total_cost_delta_vs_ma1"].mean()),
                "avg_total_delta_vs_ma2_base": float(comp["total_cost_delta_vs_ma2_base"].mean()),
                "wins_vs_waterfall": int((comp["total_cost_delta_vs_waterfall"] < 0).sum()),
                "wins_vs_oracle": int((comp["total_cost_delta_vs_oracle"] < 0).sum()),
                "wins_vs_ma1": int((comp["total_cost_delta_vs_ma1"] < 0).sum()),
                "wins_vs_ma2_base": int((comp["total_cost_delta_vs_ma2_base"] < 0).sum()),
                "bess_target_rate": float(comp[f"target_met_all_bess__{policy_name}"].mean()),
                "ev1_target_rate": float(comp[f"target_met_all_ev1__{policy_name}"].mean()),
                "ev2_target_rate": float(comp[f"target_met_all_ev2__{policy_name}"].mean()),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["avg_total_delta_vs_oracle", "avg_total_delta_vs_waterfall", "avg_total_cost"]
    )

    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"ma2_grid_summary_{args.run_tag}.csv"
    summary.to_csv(out_path, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)
    print(f"\nSummary saved to: {out_path}", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
