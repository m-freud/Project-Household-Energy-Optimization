import sqlite3

from src.simulation._sim_utils.save_results import (
    DEFAULT_HISTORY_MEASUREMENTS,
    load_history_payload_to_sqlite,
    load_results_payload_to_sqlite,
)


def test_history_writer_migrates_existing_player_id_table():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE net_load (
            player_id INTEGER,
            scenario TEXT,
            policy TEXT,
            period INTEGER,
            value REAL,
            run_id TEXT
        )
        """
    )
    connection.execute(
        """
        INSERT INTO net_load (player_id, scenario, policy, period, value, run_id)
        VALUES (7, 'old_scenario', 'old_policy', 1, 1.5, 'old_run')
        """
    )

    load_history_payload_to_sqlite(
        connection,
        {
            "household_id": 8,
            "policy": "new_policy",
            "scenario": "new_scenario",
            "run_id": "new_run",
            "history": {"net_load": [(1, 2.5)]},
        },
    )

    rows = connection.execute(
        """
        SELECT household_id, scenario, policy, period, value, run_id
        FROM net_load ORDER BY period, household_id
        """
    ).fetchall()
    assert rows == [
        (7, "old_scenario", "old_policy", 1, 1.5, "old_run"),
        (8, "new_scenario", "new_policy", 1, 2.5, "new_run"),
    ]

    for measurement in DEFAULT_HISTORY_MEASUREMENTS:
        columns = {
            row[1]
            for row in connection.execute(
                f'PRAGMA table_info("{measurement}")'
            ).fetchall()
        }
        assert {"household_id", "run_id"} <= columns


def test_results_writer_migrates_existing_player_id_table():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE results (
            run_id TEXT,
            policy TEXT,
            scenario TEXT,
            player_id INTEGER,
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
    connection.execute(
        """
        INSERT INTO results (run_id, policy, scenario, player_id)
        VALUES ('old_run', 'old_policy', 'old_scenario', 7)
        """
    )

    load_results_payload_to_sqlite(
        connection,
        {
            "run_id": "new_run",
            "policy": "new_policy",
            "scenario": "new_scenario",
            "household_id": 8,
            "has_pv": False,
            "has_bess": True,
            "total_cost": 10.0,
            "total_consumption": 20.0,
            "net_cost": 9.0,
            "net_load": 18.0,
            "target_met_bess": True,
            "target_met_ev1": True,
            "target_met_ev2": True,
            "target_met_all_bess": True,
            "target_met_all_ev1": True,
            "target_met_all_ev2": True,
            "soc_at_deadline_bess": 5.0,
            "soc_at_deadline_ev1": 6.0,
            "soc_at_deadline_ev2": 7.0,
        },
    )

    rows = connection.execute(
        "SELECT household_id, policy FROM results ORDER BY household_id"
    ).fetchall()
    assert rows == [(7, "old_policy"), (8, "new_policy")]
