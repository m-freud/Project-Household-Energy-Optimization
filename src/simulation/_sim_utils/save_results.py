import sqlite3
from typing import Any

from src.simulation.household import Household


DEFAULT_HISTORY_MEASUREMENTS = [
    "net_load",
    "net_cost",
    "total_consumption",
    "total_cost",
    "bess_soc",
    "bess_power",
    "ev1_soc",
    "ev1_power",
    "ev2_soc",
    "ev2_power",
]


def load_household_results_to_sqlite(
    results_conn: sqlite3.Connection,
    household: Household,
    policy_name: str = "no_control",
    scenario_name: str = "default_scenario",
    run_id: str | None = None,
) -> None:
    payload = {
        "run_id": run_id,
        "policy": policy_name,
        "scenario": scenario_name,
        "player_id": household.player_id,
        "has_pv": household.has_pv,
        "has_bess": household.has_bess,
        "total_cost": household.total_cost,
        "total_consumption": household.total_consumption,
        "net_cost": sum(household.history["net_cost"].values()) * 0.25,
        "net_load": sum(household.history["net_load"].values()) * 0.25,
        "target_met_bess": household.has_met_final_target("bess"),
        "target_met_ev1": household.has_met_final_target("ev1"),
        "target_met_ev2": household.has_met_final_target("ev2"),
        "target_met_all_bess": household.has_met_all_targets("bess"),
        "target_met_all_ev1": household.has_met_all_targets("ev1"),
        "target_met_all_ev2": household.has_met_all_targets("ev2"),
        "soc_at_deadline_bess": household.soc_at_deadline("bess"),
        "soc_at_deadline_ev1": household.soc_at_deadline("ev1"),
        "soc_at_deadline_ev2": household.soc_at_deadline("ev2"),
    }

    load_results_payload_to_sqlite(results_conn, payload)


def load_household_history_to_sqlite(
    results_conn: sqlite3.Connection,
    household: Household,
    policy_name: str,
    scenario_name: str,
    measurements: list[str] | None = None,
    run_id: str | None = None,
) -> None:
    if measurements is None:
        measurements = DEFAULT_HISTORY_MEASUREMENTS

    payload = {
        "run_id": run_id,
        "policy": policy_name,
        "scenario": scenario_name,
        "player_id": household.player_id,
        "history": {
            measurement: list(household.history[measurement].items())
            for measurement in measurements
        },
    }

    load_history_payload_to_sqlite(results_conn, payload)


def load_results_payload_to_sqlite(
    results_conn: sqlite3.Connection,
    payload: dict[str, Any],
) -> None:
    cursor = results_conn.cursor()
    cursor.execute(
        """
        INSERT INTO results (
            run_id, policy, scenario, player_id, has_pv, has_bess, total_cost,
            total_consumption, net_cost, net_load, target_met_bess,
            target_met_ev1, target_met_ev2, target_met_all_bess,
            target_met_all_ev1, target_met_all_ev2, soc_at_deadline_bess,
            soc_at_deadline_ev1, soc_at_deadline_ev2
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload["run_id"],
            payload["policy"],
            payload["scenario"],
            payload["player_id"],
            payload["has_pv"],
            payload["has_bess"],
            payload["total_cost"],
            payload["total_consumption"],
            payload["net_cost"],
            payload["net_load"],
            payload["target_met_bess"],
            payload["target_met_ev1"],
            payload["target_met_ev2"],
            payload["target_met_all_bess"],
            payload["target_met_all_ev1"],
            payload["target_met_all_ev2"],
            payload["soc_at_deadline_bess"],
            payload["soc_at_deadline_ev1"],
            payload["soc_at_deadline_ev2"],
        ),
    )
    results_conn.commit()


def load_history_payload_to_sqlite(
    results_conn: sqlite3.Connection,
    payload: dict[str, Any],
) -> None:
    cursor = results_conn.cursor()
    policy_name = payload["policy"]
    scenario_name = payload["scenario"]
    player_id = payload["player_id"]
    history = payload["history"]

    for measurement in DEFAULT_HISTORY_MEASUREMENTS:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {measurement} (
                player_id INTEGER,
                scenario TEXT,
                policy TEXT,
                period INTEGER,
                value REAL,
                run_id TEXT
            )
            """
        )
        table_columns = {
            row[1]
            for row in cursor.execute(
                f"PRAGMA table_info({measurement})"
            ).fetchall()
        }
        if "run_id" not in table_columns:
            cursor.execute(
                f"ALTER TABLE {measurement} ADD COLUMN run_id TEXT"
            )

        rows = history.get(measurement, [])
        if not rows:
            continue

        cursor.executemany(
            f"""
            INSERT INTO {measurement} (
                player_id, scenario, policy, period, value, run_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    player_id,
                    scenario_name,
                    policy_name,
                    int(period),
                    float(value),
                    payload.get("run_id"),
                )
                for period, value in rows
            ],
        )

    results_conn.commit()