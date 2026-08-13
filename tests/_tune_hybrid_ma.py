from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Keep imports working when script is run directly from terminal.
repo_root = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").exists()), None)
if repo_root is None:
    raise RuntimeError("Could not find repository root containing 'src'")
sys.path.insert(0, str(repo_root))

from src.config import Config
from simulation.controllers.mpc.predictors.history_avg.history_avg_predictor import HybridRunningAvgPredictor
from src.simulation.run_context import RunContext
from src.simulation.scenarios.scenario import scenarios as scenario_catalog
from src.simulation.simulation import Simulation, make_mpc_controller


@dataclass(frozen=True)
class HybridMAConfig:
    window_size: int
    persistence_range: int
    conf_interval_frct: float


def parse_int_list(raw: str) -> list[int]:
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def parse_csv(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def parse_households(raw: str) -> list[int]:
    values: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            lo, hi = sorted((start, end))
            values.update(range(lo, hi + 1))
        else:
            values.add(int(token))

    households = sorted(value for value in values if value > 0)
    if not households:
        raise ValueError("No valid household IDs parsed from --households")
    return households


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def next_run_id(conn: sqlite3.Connection) -> int:
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


def scenario_lookup() -> dict[str, object]:
    return {scenario.name: scenario for scenario in scenario_catalog}


def select_scenarios(scenario_names_csv: str | None) -> list[object]:
    all_by_name = scenario_lookup()
    if not scenario_names_csv:
        return list(scenario_catalog)

    selected_names = parse_csv(scenario_names_csv)
    missing = [name for name in selected_names if name not in all_by_name]
    if missing:
        known = ", ".join(sorted(all_by_name))
        raise ValueError(f"Unknown scenarios: {missing}. Known scenarios: {known}")

    return [all_by_name[name] for name in selected_names]


def build_configs(args: argparse.Namespace) -> list[HybridMAConfig]:
    window_sizes = parse_int_list(args.window_sizes)
    if not window_sizes:
        raise ValueError("--window-sizes must contain at least one value")

    configs: list[HybridMAConfig] = []

    for window_size in window_sizes:
        window_size = max(1, int(window_size))
        persistence_range = max(0, int(args.persistence_range))
        conf_interval_frct = 0.0 if args.preset == "short_only" else float(args.conf_interval_frct)
        configs.append(
            HybridMAConfig(
                window_size=window_size,
                persistence_range=persistence_range,
                conf_interval_frct=conf_interval_frct,
            )
        )

    return configs


def build_policy_name(config: HybridMAConfig, run_tag: str) -> str:
    return (
        f"mpc_hybrid_ma_w{config.window_size}"
        f"_pr{config.persistence_range}"
        f"_ci{config.conf_interval_frct:.2f}"
        f"_{run_tag}"
    )


def config_to_row(config: HybridMAConfig) -> dict[str, int | float | str]:
    return {
        "window_size": config.window_size,
        "persistence_range": config.persistence_range,
        "conf_interval_frct": config.conf_interval_frct,
    }


def run_config(
    sim: Simulation,
    config: HybridMAConfig,
    scenarios: list[object],
    households: list[int],
    workers: int | None,
    run_id: str,
    run_tag: str,
) -> tuple[str, str]:
    policy_name = build_policy_name(config, run_tag)

    predictor = HybridRunningAvgPredictor(
        conf_interval_frct=config.conf_interval_frct,
    )

    controller_factory = make_mpc_controller(
        policy_name,
        horizon=96,
        predictor=predictor,
    )

    run_contexts = [
        RunContext(
            controller_factory=controller_factory,
            controller_name=policy_name,
            scenario=scenario,
            run_id=run_id,
            start_time=1,
        )
        for scenario in scenarios
    ]

    print(
        f"\nRunning {policy_name} | run_id={run_id} | "
        f"households={len(households)} scenarios={len(scenarios)}",
        flush=True,
    )

    sim.run_batch(
        run_contexts,
        household_ids=households,
        parallel_households=True,
        parallel_workers=workers,
    )

    return policy_name, run_id


def build_reports(
    db_path: Path,
    executed_runs: list[dict[str, object]],
    households: list[int],
    scenarios: list[object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not executed_runs:
        return pd.DataFrame(), pd.DataFrame()

    scenario_names = [scenario.name for scenario in scenarios]
    policy_to_run = {str(run["policy"]): str(run["run_id"]) for run in executed_runs}
    metadata = pd.DataFrame(executed_runs)

    conn = sqlite3.connect(db_path)
    try:
        results = pd.read_sql_query(
            """
            SELECT run_id, policy, scenario, player_id,
                   total_cost, net_cost, net_load,
                   target_met_all_bess, target_met_all_ev1, target_met_all_ev2
            FROM results
            """,
            conn,
        )
    finally:
        conn.close()

    scoped = results[
        results["player_id"].isin(households)
        & results["scenario"].isin(scenario_names)
        & results["policy"].isin(policy_to_run)
    ].copy()

    if scoped.empty:
        return pd.DataFrame(), pd.DataFrame()

    scoped = scoped[scoped.apply(lambda row: str(row["run_id"]) == policy_to_run.get(str(row["policy"]), ""), axis=1)]

    detail = scoped.groupby(["policy", "run_id", "scenario"], as_index=False).agg(
        pairs=("player_id", "count"),
        avg_total_cost=("total_cost", "mean"),
        avg_net_cost=("net_cost", "mean"),
        avg_net_load=("net_load", "mean"),
        bess_target_rate=("target_met_all_bess", "mean"),
        ev1_target_rate=("target_met_all_ev1", "mean"),
        ev2_target_rate=("target_met_all_ev2", "mean"),
    )
    detail = detail.merge(metadata, on=["policy", "run_id"], how="left")

    summary = scoped.groupby(["policy", "run_id"], as_index=False).agg(
        pairs=("player_id", "count"),
        avg_total_cost=("total_cost", "mean"),
        avg_net_cost=("net_cost", "mean"),
        avg_net_load=("net_load", "mean"),
        bess_target_rate=("target_met_all_bess", "mean"),
        ev1_target_rate=("target_met_all_ev1", "mean"),
        ev2_target_rate=("target_met_all_ev2", "mean"),
    )
    summary = summary.merge(metadata, on=["policy", "run_id"], how="left")

    summary = summary.sort_values(["avg_net_cost", "avg_total_cost", "window_size"]).reset_index(drop=True)
    detail = detail.sort_values(["window_size", "scenario"]).reset_index(drop=True)
    return summary, detail


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid MA hyperparameter tuning runner")

    parser.add_argument(
        "--preset",
        choices=["short_only", "manual"],
        default="manual",
        help="short_only disables all non-short-window effects; manual uses explicit args",
    )

    parser.add_argument(
        "--window-sizes",
        default="48,96",
        help="Comma-separated window sizes to evaluate (e.g. 48,96)",
    )

    parser.add_argument("--conf-interval-frct", type=float, default=0)
    parser.add_argument(
        "--persistence-range",
        type=int,
        default=1,
        help="Number of initial forecast steps to persist the latest observed value",
    )

    parser.add_argument(
        "--households",
        default="1-48",
        help="Household IDs as CSV/ranges (e.g. 1-48 or 1,2,10-20)",
    )
    parser.add_argument(
        "--scenarios",
        default=None,
        help="Comma-separated scenario names. Omit to run all scenarios.",
    )
    parser.add_argument("--parallel-workers", type=int, default=6)
    parser.add_argument("--run-tag", default="hybrid_ma_tune")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    households = parse_households(args.households)
    scenarios = select_scenarios(args.scenarios)
    configs = build_configs(args)

    print("Starting Hybrid MA tuning", flush=True)
    print(f"Preset: {args.preset}", flush=True)
    print(f"Window sizes : {[cfg.window_size for cfg in configs]}", flush=True)
    print(f"Grid size    : {len(configs)}", flush=True)
    print(f"Households ({len(households)}): {households[:8]}{'...' if len(households) > 8 else ''}", flush=True)
    print(f"Scenarios ({len(scenarios)}): {[s.name for s in scenarios]}", flush=True)

    sim_conn = sqlite3.connect(Config.SQLITE_PATH)
    executed_runs: list[dict[str, object]] = []
    total_configs = len(configs)
    sweep_start = time.perf_counter()

    try:
        sim = Simulation(sim_conn)
        run_id_counter = next_run_id(sim_conn)

        for index, config in enumerate(configs, start=1):
            run_start = time.perf_counter()
            run_id = str(run_id_counter)
            run_id_counter += 1
            policy_name, assigned_run_id = run_config(
                sim=sim,
                config=config,
                scenarios=scenarios,
                households=households,
                workers=args.parallel_workers,
                run_id=run_id,
                run_tag=str(args.run_tag),
            )
            run_record: dict[str, object] = {
                "policy": policy_name,
                "run_id": assigned_run_id,
                "run_tag": str(args.run_tag),
                "preset": str(args.preset),
                "household_count": len(households),
                "scenario_count": len(scenarios),
            }
            run_record.update(config_to_row(config))
            executed_runs.append(run_record)

            run_elapsed = time.perf_counter() - run_start
            elapsed = time.perf_counter() - sweep_start
            avg_per_config = elapsed / float(index)
            remaining_configs = total_configs - index
            eta_seconds = avg_per_config * float(remaining_configs)
            est_total = avg_per_config * float(total_configs)
            progress_pct = 100.0 * float(index) / float(total_configs)

            print(
                (
                    f"Progress: {index}/{total_configs} ({progress_pct:.1f}%) | "
                    f"last={format_duration(run_elapsed)} | "
                    f"elapsed={format_duration(elapsed)} | "
                    f"eta={format_duration(eta_seconds)} | "
                    f"est_total={format_duration(est_total)}"
                ),
                flush=True,
            )
    finally:
        sim_conn.close()

    summary_df, detail_df = build_reports(
        db_path=Path(Config.SQLITE_PATH),
        executed_runs=executed_runs,
        households=households,
        scenarios=scenarios,
    )

    reports_dir = repo_root / "reports" / "tuning"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / f"hybrid_ma_tuning_summary_{args.run_tag}.csv"
    detail_path = reports_dir / f"hybrid_ma_tuning_detail_{args.run_tag}.csv"

    summary_df.to_csv(summary_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 80)

    print("\nTuning complete.", flush=True)
    print(f"Summary CSV: {summary_path}", flush=True)
    print(f"Detail CSV : {detail_path}", flush=True)
    if not summary_df.empty:
        display_columns = [
            "window_size",
            "avg_net_cost",
            "avg_total_cost",
            "pairs",
            "bess_target_rate",
            "ev1_target_rate",
            "ev2_target_rate",
            "policy",
            "run_id",
        ]
        available_columns = [column for column in display_columns if column in summary_df.columns]
        print("\nRanked summary (lower avg_net_cost is better):", flush=True)
        print(summary_df[available_columns].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
