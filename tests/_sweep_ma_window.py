from __future__ import annotations

import sqlite3
from pathlib import Path
import sys

import pandas as pd

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.sqlite_connection import create_sqlite_connection
from src.simulation.run_context import RunContext
from src.simulation.scenarios.scenario import scenarios as scenario_catalog
from src.simulation.simulation import Simulation, make_mpc_controller
from src.simulation.controllers.mpc.predictors.moving_average_predictor import MovingAveragePredictor


HOUSEHOLDS = list(range(1, 21))
WINDOW_SIZES = [2, 4, 6, 8, 12, 16, 24, 32, 48]
BASELINE_RUN_ID = "1"
BASELINE_POLICIES = ["waterfall", "mpc_oracle"]


def _next_run_id(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """
        SELECT COALESCE(MAX(CAST(run_id AS INTEGER)), 0)
        FROM results
        WHERE run_id IS NOT NULL
          AND TRIM(run_id) <> ''
          AND run_id NOT GLOB '*[^0-9]*'
        """
    ).fetchone()
    return int(row[0] or 0) + 1


def run_sweep() -> tuple[pd.DataFrame, pd.DataFrame]:
    sim_conn = create_sqlite_connection()
    try:
        sim = Simulation(sim_conn)
        run_id_counter = _next_run_id(sim_conn)

        for window_size in WINDOW_SIZES:
            run_id = str(run_id_counter)
            run_id_counter += 1
            policy_name = f"mpc_ma_w{window_size}"

            print(f"Running window_size={window_size} with run_id={run_id} ...")

            controller_factory = make_mpc_controller(
                policy_name,
                horizon=96,
                predictor=MovingAveragePredictor(window_size=window_size),
            )

            run_contexts = [
                RunContext(
                    controller_factory=controller_factory,
                    controller_name=policy_name,
                    scenario=scenario,
                    run_id=run_id,
                    start_time=1,
                )
                for scenario in scenario_catalog
            ]

            sim.run_batch(
                run_contexts,
                household_ids=HOUSEHOLDS,
                max_households=None,
                parallel_households=True,
                parallel_workers=None,
            )

        analysis_conn = sqlite3.connect(Config.SQLITE_PATH)
        try:
            result_df = pd.read_sql_query(
                """
                SELECT run_id, policy, scenario, player_id,
                       total_cost, net_cost, net_load,
                       target_met_all_bess, target_met_all_ev1, target_met_all_ev2
                FROM results
                WHERE player_id BETWEEN 1 AND 20
                """,
                analysis_conn,
            )
        finally:
            analysis_conn.close()

    finally:
        sim_conn.close()

    sweep_policies = [f"mpc_ma_w{w}" for w in WINDOW_SIZES]
    selected = result_df[
        ((result_df["policy"].isin(BASELINE_POLICIES)) & (result_df["run_id"] == BASELINE_RUN_ID))
        | (result_df["policy"].isin(sweep_policies))
    ].copy()

    rows: list[dict] = []
    detail_rows: list[dict] = []

    for window_size in WINDOW_SIZES:
        policy_name = f"mpc_ma_w{window_size}"
        ma_df = selected[selected["policy"] == policy_name].copy()

        if ma_df.empty:
            continue

        # Keep only newest sweep run for this policy.
        run_ids = ma_df["run_id"].dropna().astype(str)
        latest_run_id = str(max(int(rid) for rid in run_ids if rid.isdigit()))

        subset = selected[
            ((selected["policy"].isin(BASELINE_POLICIES)) & (selected["run_id"] == BASELINE_RUN_ID))
            | ((selected["policy"] == policy_name) & (selected["run_id"] == latest_run_id))
        ].copy()

        counts = subset.groupby(["scenario", "player_id"])["policy"].nunique().reset_index(name="n")
        valid = counts[counts["n"] == 3][["scenario", "player_id"]]
        subset = subset.merge(valid, on=["scenario", "player_id"], how="inner")

        pivot = subset.pivot_table(
            index=["scenario", "player_id"],
            columns="policy",
            values=["total_cost", "net_cost", "net_load", "target_met_all_bess", "target_met_all_ev1", "target_met_all_ev2"],
            aggfunc="first",
        )

        pivot.columns = [f"{metric}__{policy}" for metric, policy in pivot.columns]
        comp = pivot.reset_index()

        for metric in ["total_cost", "net_cost", "net_load"]:
            comp[f"{metric}_delta_vs_waterfall"] = comp[f"{metric}__{policy_name}"] - comp[f"{metric}__waterfall"]
            comp[f"{metric}_delta_vs_oracle"] = comp[f"{metric}__{policy_name}"] - comp[f"{metric}__mpc_oracle"]

        detail_rows.extend(
            {
                "window_size": window_size,
                "run_id": latest_run_id,
                "scenario": row["scenario"],
                "player_id": int(row["player_id"]),
                "total_cost_ma": row[f"total_cost__{policy_name}"],
                "total_cost_waterfall": row["total_cost__waterfall"],
                "total_cost_oracle": row["total_cost__mpc_oracle"],
                "delta_total_vs_waterfall": row["total_cost_delta_vs_waterfall"],
                "delta_total_vs_oracle": row["total_cost_delta_vs_oracle"],
            }
            for _, row in comp.iterrows()
        )

        rows.append(
            {
                "window_size": window_size,
                "run_id": latest_run_id,
                "pairs": int(comp.shape[0]),
                "avg_total_cost": float(comp[f"total_cost__{policy_name}"].mean()),
                "avg_total_delta_vs_waterfall": float(comp["total_cost_delta_vs_waterfall"].mean()),
                "avg_total_delta_vs_oracle": float(comp["total_cost_delta_vs_oracle"].mean()),
                "avg_net_delta_vs_waterfall": float(comp["net_cost_delta_vs_waterfall"].mean()),
                "avg_net_delta_vs_oracle": float(comp["net_cost_delta_vs_oracle"].mean()),
                "wins_vs_waterfall": int((comp["total_cost_delta_vs_waterfall"] < 0).sum()),
                "wins_vs_oracle": int((comp["total_cost_delta_vs_oracle"] < 0).sum()),
                "bess_target_rate": float(comp[f"target_met_all_bess__{policy_name}"].mean()),
                "ev1_target_rate": float(comp[f"target_met_all_ev1__{policy_name}"].mean()),
                "ev2_target_rate": float(comp[f"target_met_all_ev2__{policy_name}"].mean()),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("avg_total_delta_vs_oracle")
    detail_df = pd.DataFrame(detail_rows)
    return summary_df, detail_df


def main() -> None:
    summary_df, detail_df = run_sweep()

    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_path = report_dir / "ma_window_sweep_summary.csv"
    detail_path = report_dir / "ma_window_sweep_detail.csv"

    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 50)

    print("\nSensitivity sweep complete.")
    print(f"Summary CSV: {summary_path}")
    print(f"Detail CSV : {detail_path}")
    print("\nRanked summary (lower avg_total_delta_vs_oracle is better):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
