"""
Split-timing profile for one MPC household run (MA3 predictor).
Reports: predictor, param-update, solve, and overhead per timestep.
"""
from __future__ import annotations

import time
from pathlib import Path
import sys

repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), Path.cwd())
sys.path.insert(0, str(repo_root))

import numpy as np
import cvxpy as cp
from types import SimpleNamespace

from src.simulation.simulation import Simulation
from src.simulation.scenarios.scenario import default_scenario
from src.simulation.controllers.mpc.mpc_controller import MPCController
from simulation.controllers.mpc.predictors.history_avg.history_avg_predictor import HybridRunningAvgPredictor
from src.sqlite_connection import create_sqlite_connection

PLAYER_ID = 1


def main():
    connection = create_sqlite_connection()
    try:
        sim = Simulation(connection, ensure_schema=False)
        scenario = default_scenario
        run_context = SimpleNamespace(scenario=scenario, start_time=1)
        household = sim.create_household(PLAYER_ID, run_context)

        predictor = HybridRunningAvgPredictor(conf_interval_frct=0.0)

        controller = MPCController(
            name="profile_history_avg",
            household=household,
            scenario=scenario,
            horizon=96,
            predictor=predictor,
            duration_hours=sim.duration_hours,
        )

        t_predict = 0.0
        t_params = 0.0
        t_solve = 0.0
        t_other = 0.0
        n = 0

        # Monkey-patch set_controls to capture split timing
        original_set_controls = controller.set_controls

        def timed_set_controls(hh, sc, **kwargs):
            nonlocal t_predict, t_params, t_solve, t_other, n

            current_timestep = hh.current_timestep
            planning_horizon = controller._planning_horizon(current_timestep)
            controller._ensure_compiled_problem(planning_horizon)

            # --- predictor ---
            t0 = time.perf_counter()
            predictions = predictor.predict(hh, sc, planning_horizon)
            t1 = time.perf_counter()
            t_predict += t1 - t0

            # --- param updates (everything between predict and solve) ---
            # We call the real set_controls but intercept solve via the problem object
            problem = controller._problem

            t2 = time.perf_counter()
            # Inline the param-fill by calling original but we need to intercept solve.
            # Easier: call original, which does predict+params+solve internally,
            # but we already have predict time above.  Patch problem.solve instead.

            solve_times: list[float] = []
            real_solve = problem.solve

            def timed_solve(*a, **kw):
                ts = time.perf_counter()
                result = real_solve(*a, **kw)
                solve_times.append(time.perf_counter() - ts)
                return result

            problem.solve = timed_solve
            result = original_set_controls(hh, sc, **kwargs)
            problem.solve = real_solve
            t3 = time.perf_counter()

            total = t3 - t0
            solve_t = solve_times[0] if solve_times else 0.0
            param_t = total - (t1 - t0) - solve_t

            t_params += param_t
            t_solve += solve_t
            n += 1
            return result

        controller.set_controls = timed_set_controls

        # Run full simulation
        wall_start = time.perf_counter()
        for t in range(1, sim.num_timesteps + 1):
            sim.step(household, controller, scenario, duration_hours=sim.duration_hours, time=t)
        wall_total = time.perf_counter() - wall_start

        print(f"\n=== MPC Split Timing: player={PLAYER_ID}, predictor=HybridMA, steps={n} ===")
        print(f"  Total wall time : {wall_total:.3f}s")
        print(f"  Predictor       : {t_predict:.3f}s  ({100*t_predict/wall_total:.1f}%)")
        print(f"  Param update    : {t_params:.3f}s  ({100*t_params/wall_total:.1f}%)")
        print(f"  Solve (CVXPY)   : {t_solve:.3f}s  ({100*t_solve/wall_total:.1f}%)")
        print(f"  Other/sim       : {wall_total - t_predict - t_params - t_solve:.3f}s")
        print(f"  Avg per step    : {wall_total/n*1000:.1f}ms  (solve: {t_solve/n*1000:.1f}ms, predict: {t_predict/n*1000:.1f}ms)")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
