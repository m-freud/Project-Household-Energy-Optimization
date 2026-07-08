from __future__ import annotations

import argparse
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
import sys

repo_root = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").exists()), None)
if repo_root is None:
    raise RuntimeError("Could not find repository root containing 'src'")
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.simulation.controllers.mpc.config.device_buffer_config import DeviceBufferConfig
from src.simulation.run_context import RunContext
from src.simulation.scenarios.scenario import scenarios as scenario_catalog
from src.simulation.simulation import Simulation, make_mpc_controller


def _parse_buffer_steps_list(raw_value: str) -> list[int]:
    parsed: list[int] = []
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        parsed.append(max(0, int(token)))
    if not parsed:
        parsed = [0, 4, 8]
    return sorted(set(parsed))


def _clear_policy_tables(cur: sqlite3.Cursor) -> None:
    tables = [row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    for table in tables:
        cols = [row[1] for row in cur.execute(f"PRAGMA table_info({table})").fetchall()]
        if "policy" in cols:
            cur.execute(f"DELETE FROM {table}")


def _run_batch(sim: Simulation, run_contexts: list[RunContext], households: list[int], workers: int) -> None:
    for idx, run_context in enumerate(run_contexts, start=1):
        print(f"[{idx}/{len(run_contexts)}] {run_context.controller_name} / {run_context.scenario.name} ...", flush=True)
        sim.run_all_households(
            run_context,
            household_ids=households,
            parallel_households=True,
            parallel_workers=workers,
        )


def _print_summary(cur: sqlite3.Cursor, policy_name: str) -> None:
    overall = cur.execute(
        """
        SELECT
            COUNT(*) AS rows_n,
            AVG(total_cost) AS avg_total_cost,
            AVG(net_cost) AS avg_net_cost,
            SUM(CASE WHEN target_met_all_bess = 1 AND target_met_all_ev1 = 1 AND target_met_all_ev2 = 1 THEN 1 ELSE 0 END) AS all_targets_ok
        FROM results
        WHERE policy = ?
        """,
        (policy_name,),
    ).fetchone()

    print(
        f"{policy_name}: rows={overall[0]}, avg_total_cost={overall[1]:.4f}, "
        f"avg_net_cost={overall[2]:.4f}, all_targets_ok={overall[3]}/{overall[0]}"
    )


def _print_comparison(cur: sqlite3.Cursor, baseline_policy: str, compare_policy: str, household_count: int) -> None:
    by_scenario = cur.execute(
        """
        SELECT
            b.scenario,
            AVG(b.total_cost) AS base_avg_total_cost,
            AVG(t.total_cost) AS cmp_avg_total_cost,
            AVG(t.total_cost) - AVG(b.total_cost) AS delta_total_cost,
            AVG(b.net_cost) AS base_avg_net_cost,
            AVG(t.net_cost) AS cmp_avg_net_cost,
            AVG(t.net_cost) - AVG(b.net_cost) AS delta_net_cost,
            SUM(CASE WHEN b.target_met_all_bess = 1 AND b.target_met_all_ev1 = 1 AND b.target_met_all_ev2 = 1 THEN 1 ELSE 0 END) AS base_all_targets_ok,
            SUM(CASE WHEN t.target_met_all_bess = 1 AND t.target_met_all_ev1 = 1 AND t.target_met_all_ev2 = 1 THEN 1 ELSE 0 END) AS cmp_all_targets_ok
        FROM results b
        JOIN results t
          ON t.player_id = b.player_id
         AND t.scenario = b.scenario
        WHERE b.policy = ?
          AND t.policy = ?
        GROUP BY b.scenario
        ORDER BY b.scenario
        """,
        (baseline_policy, compare_policy),
    ).fetchall()

    print(f"\nScenario comparison ({compare_policy} - {baseline_policy}):")
    for row in by_scenario:
        print(
            f"  {row[0]}: d_total={row[3]:+.4f}, d_net={row[6]:+.4f}, "
            f"targets base={row[7]}/{household_count} cmp={row[8]}/{household_count}"
        )

    overall = cur.execute(
        """
        SELECT
            AVG(t.total_cost) - AVG(b.total_cost) AS delta_total_cost,
            AVG(t.net_cost) - AVG(b.net_cost) AS delta_net_cost,
            SUM(CASE WHEN b.target_met_all_bess = 1 AND b.target_met_all_ev1 = 1 AND b.target_met_all_ev2 = 1 THEN 1 ELSE 0 END) AS base_all_targets_ok,
            SUM(CASE WHEN t.target_met_all_bess = 1 AND t.target_met_all_ev1 = 1 AND t.target_met_all_ev2 = 1 THEN 1 ELSE 0 END) AS cmp_all_targets_ok,
            COUNT(*) AS n
        FROM results b
        JOIN results t
          ON t.player_id = b.player_id
         AND t.scenario = b.scenario
        WHERE b.policy = ?
          AND t.policy = ?
        """,
        (baseline_policy, compare_policy),
    ).fetchone()

    print(f"\nOverall comparison ({compare_policy} - {baseline_policy}):")
    print(
        f"  d_total={overall[0]:+.4f}, d_net={overall[1]:+.4f}, "
        f"targets base={overall[2]}/{overall[4]} cmp={overall[3]}/{overall[4]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick oracle multi-buffer sweep for visual dashboard comparisons")
    parser.add_argument("--households", type=int, default=5, help="Number of households starting from ID 1")
    parser.add_argument("--buffer-steps", type=int, default=4, help="Backward-compatible single buffered run value")
    parser.add_argument(
        "--buffer-steps-list",
        type=str,
        default="0,4,8",
        help="Comma-separated list of buffer steps (e.g. 0,4,8). When provided, this takes precedence.",
    )
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source_db = root / "sqlite" / "energy.db"
    backup_dir = root / "sqlite" / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_db = backup_dir / f"quick_oracle_tb4_compare_{timestamp}.db"
    shutil.copy2(source_db, out_db)

    conn = sqlite3.connect(out_db)
    cur = conn.cursor()
    _clear_policy_tables(cur)
    conn.commit()

    Config.SQLITE_PATH = out_db

    sim = Simulation(conn, ensure_schema=False)
    household_count = max(1, int(args.households))
    households = list(range(1, household_count + 1))
    buffer_steps_list = _parse_buffer_steps_list(args.buffer_steps_list)
    if not buffer_steps_list:
        buffer_steps_list = [0, max(0, int(args.buffer_steps))]
    workers = max(1, int(args.workers))

    policy_by_buffer = {steps: f"mpc_oracle_tb{steps}" for steps in buffer_steps_list}

    print(f"output_db={out_db}")
    print(f"households={households} scenarios={len(scenario_catalog)} workers={workers} buffer_steps_list={buffer_steps_list}")

    for steps in buffer_steps_list:
        policy_name = policy_by_buffer[steps]
        run_label = f"quick_tb{steps}"
        factory = make_mpc_controller(
            policy_name,
            horizon=96,
            duration_hours=sim.duration_hours,
            buffer_config=DeviceBufferConfig.with_universal_time_buffer(steps),
        )
        contexts = [
            RunContext(
                controller_factory=factory,
                controller_name=policy_name,
                scenario=scenario,
                run_id=run_label,
                start_time=1,
            )
            for scenario in scenario_catalog
        ]
        print(f"\nRunning policy={policy_name} (time buffer = {steps} steps)...")
        _run_batch(sim, contexts, households, workers)

    baseline_policy = policy_by_buffer[0] if 0 in policy_by_buffer else policy_by_buffer[buffer_steps_list[0]]
    for steps in buffer_steps_list:
        _print_summary(cur, policy_by_buffer[steps])

    for steps in buffer_steps_list:
        compare_policy = policy_by_buffer[steps]
        if compare_policy == baseline_policy:
            continue
        _print_comparison(cur, baseline_policy, compare_policy, household_count)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
