"""Full-horizon oracle benchmark for a single household/scenario pair.

This solves the MPC model once at t=1 with perfect future inputs for the full
96-step day, then compares the resulting total cost against stored rolling
oracle and MA2 baselines.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

# Keep imports working when the script is run directly from terminal.
repo_root = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").exists()), None)
if repo_root is None:
    raise RuntimeError("Could not find repository root containing 'src'")
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.simulation.controllers.mpc.mpc_controller import MPCController
from src.simulation.controllers.mpc.predictors.moving_average_predictor2 import MovingAveragePredictor2
from simulation.controllers.mpc.predictors.oracle.oracle_predictor import OraclePredictor
from src.simulation.run_context import RunContext
from src.simulation.scenarios.scenario import scenarios as scenario_catalog
from src.simulation.simulation import Simulation, make_mpc_controller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve a full-horizon oracle benchmark and compare it to stored baselines.")
    parser.add_argument("--household", type=int, default=15)
    parser.add_argument("--scenario", default="default_scenario")
    parser.add_argument("--oracle-policy", default="mpc_oracle")
    parser.add_argument("--oracle-run-id", default="1")
    parser.add_argument("--ma2-policy", default="mpc_ma2_s7_l58_ma2_longscan_56_58")
    parser.add_argument("--ma2-run-id", default="41")
    return parser.parse_args()


def load_last_row(cur: sqlite3.Cursor, policy: str, run_id: str, scenario: str, household_id: int) -> dict:
    row = cur.execute(
        """
        SELECT total_cost, net_cost, net_load
        FROM results
        WHERE policy = ? AND run_id = ? AND scenario = ? AND player_id = ?
        ORDER BY rowid DESC
        LIMIT 1
        """,
        (policy, run_id, scenario, household_id),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Missing baseline row for policy={policy}, run_id={run_id}, scenario={scenario}, household={household_id}")
    return {"total_cost": float(row[0]), "net_cost": float(row[1]), "net_load": float(row[2])}


def main() -> None:
    args = parse_args()
    scenarios_by_name = {scenario.name: scenario for scenario in scenario_catalog}
    if args.scenario not in scenarios_by_name:
        raise ValueError(f"Unknown scenario: {args.scenario}")

    scenario = scenarios_by_name[args.scenario]
    household_id = int(args.household)

    conn = sqlite3.connect(Config.SQLITE_PATH)
    try:
        sim = Simulation(conn, ensure_schema=False)

        benchmark_policy = "full_horizon_oracle_benchmark"
        benchmark_rc = RunContext(
            controller_factory=make_mpc_controller(
                benchmark_policy,
                horizon=96,
                predictor=OraclePredictor(),
            ),
            controller_name=benchmark_policy,
            scenario=scenario,
            start_time=1,
        )

        household = sim.create_household(household_id, benchmark_rc)
        controller = sim.create_controller(household, benchmark_rc)
        if not isinstance(controller, MPCController):
            raise TypeError("Benchmark controller must be an MPCController")

        # Solve the whole day once at the start of the day.
        controller.set_controls(household, scenario)

        if controller._problem is None or controller._problem.value is None:
            raise RuntimeError("Benchmark problem did not produce a valid solution")

        control_arrays = {
            "bess_power": None if controller._vars.get("bess_power") is None else np.asarray(controller._vars["bess_power"].value).reshape(-1),
            "ev1_power": None if controller._vars.get("ev1_charge") is None else np.asarray(controller._vars["ev1_charge"].value).reshape(-1),
            "ev2_power": None if controller._vars.get("ev2_charge") is None else np.asarray(controller._vars["ev2_charge"].value).reshape(-1),
        }

        replay = sim.create_household(household_id, benchmark_rc)
        benchmark_replayed_total_cost = 0.0
        for timestep in range(1, 97):
            replay.current_timestep = timestep
            sim.update_household_inputs(replay)

            step_controls: dict[str, float] = {}
            for key, values in control_arrays.items():
                if values is not None:
                    step_controls[key] = float(values[timestep - 1])

            replay.apply_controls(step_controls)
            benchmark_replayed_total_cost += replay.net_cost * 0.25

        benchmark_replayed_total_cost += replay.base_cost

        benchmark_objective_total_cost = float(controller._problem.value) + float(household.base_cost)

        cur = conn.cursor()
        rolling_oracle = load_last_row(cur, args.oracle_policy, args.oracle_run_id, scenario.name, household_id)
        ma2 = load_last_row(cur, args.ma2_policy, args.ma2_run_id, scenario.name, household_id)

        print(f"pair {scenario.name} {household_id}")
        print(f"full_horizon_objective_total_cost {benchmark_objective_total_cost:.12f}")
        print(f"full_horizon_replayed_total_cost {benchmark_replayed_total_cost:.12f}")
        print(f"full_horizon_objective_gap {benchmark_replayed_total_cost - benchmark_objective_total_cost:+.12f}")
        print(f"rolling_oracle_total_cost {rolling_oracle['total_cost']:.12f}")
        print(f"ma2_total_cost {ma2['total_cost']:.12f}")
        print(f"full_minus_rolling_oracle {benchmark_replayed_total_cost - rolling_oracle['total_cost']:+.12f}")
        print(f"full_minus_ma2 {benchmark_replayed_total_cost - ma2['total_cost']:+.12f}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()