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
        "household_id": household.household_id,
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
        "household_id": household.household_id,
        "history": {
            measurement: list(household.history[measurement].items())
            for measurement in measurements
        },
    }

    load_history_payload_to_sqlite(results_conn, payload)


def load_results_payload_to_sqlite(
    results_conn: sqlite3.Connection,
    payload: dict[str, Any],
    ensure_table: bool = False,
) -> None:
    if ensure_table:
        results_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                run_id TEXT,
                policy TEXT,
                scenario TEXT,
                household_id INTEGER,
                has_pv BOOLEAN,
                has_bess BOOLEAN,
                total_cost REAL,
                total_consumption REAL,
                net_cost REAL,
                net_load REAL,
                target_met_bess BOOLEAN,
                target_met_ev1 BOOLEAN,
                target_met_ev2 BOOLEAN,
                target_met_all_bess BOOLEAN,
                target_met_all_ev1 BOOLEAN,
                target_met_all_ev2 BOOLEAN,
                soc_at_deadline_bess REAL,
                soc_at_deadline_ev1 REAL,
                soc_at_deadline_ev2 REAL
        )
        """
    )
    cursor = results_conn.cursor()
    cursor.execute(
        """
        INSERT INTO results (
            run_id, policy, scenario, household_id, has_pv, has_bess, total_cost,
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
            payload["household_id"],
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
    household_id = payload["household_id"]
    history = payload["history"]

    for measurement in DEFAULT_HISTORY_MEASUREMENTS:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {measurement} (
                household_id INTEGER,
                scenario TEXT,
                policy TEXT,
                period INTEGER,
                value REAL,
                run_id TEXT
            )
            """
        )
        table_columns = _table_columns(results_conn, measurement)
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
                household_id, scenario, policy, period, value, run_id
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    household_id,
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


def load_forecasts_payload_to_sqlite(
    forecasts_conn: sqlite3.Connection,
    payload: dict[str, Any],
) -> None:
    forecasts = payload.get("forecast_log", {})
    histories = payload.get("history", {})

    rows = []

    for target, target_forecasts in forecasts.items():
        if isinstance(target_forecasts, dict):
            forecast_items = target_forecasts.items()
        else:
            forecast_items = target_forecasts

        if isinstance(histories, dict):
            history_values = histories.get(target)
        else:
            history_values = None

        if history_values is None:
            if target == "ev1_status":
                history_values = _ev_status_history(histories, "ev1")
            elif target == "ev2_status":
                history_values = _ev_status_history(histories, "ev2")
            elif target == "ev1_at_home":
                history_values = histories.get("ev1_at_home") if isinstance(histories, dict) else None
            elif target == "ev2_at_home":
                history_values = histories.get("ev2_at_home") if isinstance(histories, dict) else None
            elif target == "ev1_at_charging_station":
                history_values = histories.get("ev1_at_charging_station") if isinstance(histories, dict) else None
            elif target == "ev2_at_charging_station":
                history_values = histories.get("ev2_at_charging_station") if isinstance(histories, dict) else None

        actuals = _history_as_dict(history_values)

        for key, prediction in forecast_items:
            timestep, horizon = key
            actual_timestep = int(timestep) + int(horizon)
            rows.append((
                payload["run_id"],
                payload["household_id"],
                target,
                payload.get("predictor"),
                int(timestep),
                int(horizon),
                float(prediction),
                actuals.get(actual_timestep),
            ))

    if rows:
        forecasts_conn.executemany(
            """
            INSERT INTO forecasts (
                run_id, household_id, target, predictor,
                timestep, horizon, prediction, actual
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        forecasts_conn.commit()


def _history_as_dict(history: Any) -> dict[int, Any]:
    if isinstance(history, dict):
        return {int(period): value for period, value in history.items()}
    if isinstance(history, (list, tuple)):
        return {int(period): value for period, value in history}
    return {}


def _ev_status_history(histories: Any, ev_key: str) -> dict[int, int] | None:
    if not isinstance(histories, dict):
        return None

    at_home = _history_as_dict(histories.get(f"{ev_key}_at_home"))
    at_station = _history_as_dict(histories.get(f"{ev_key}_at_charging_station"))
    periods = set(at_home) | set(at_station)

    if not periods:
        return None

    return {
        period: (
            0 if at_home.get(period) == 1
            else 2 if at_station.get(period) == 1
            else 1
        )
        for period in periods
    }



def _table_columns(results_conn: sqlite3.Connection, table_name: str) -> set[str]: #TODO remove
    return {
        row[1]
        for row in results_conn.execute(
            f'PRAGMA table_info("{table_name}")'
        ).fetchall()
    }
