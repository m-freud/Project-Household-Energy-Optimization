import sqlite3
import re

import pandas as pd


# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))
from src.config import Config

def create_sqlite_connection():
    return sqlite3.connect(Config.SQLITE_PATH)


def get_sqlite_cursor():
    return create_sqlite_connection().cursor()


sqlite_conn = create_sqlite_connection()
sqlite_cursor = get_sqlite_cursor()


def _is_safe_identifier(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value))


def load_series(
    table_name: str,
    player_id: int,
    scenario_name: str | None = None,
    policy_name: str | None = None,
) -> pd.DataFrame:
    if not _is_safe_identifier(table_name):
        raise ValueError(f"Invalid table name: {table_name}")

    with sqlite3.connect(Config.SQLITE_PATH) as conn:
        try:
            table_info = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        except sqlite3.OperationalError:
            return pd.DataFrame()

        if not table_info:
            return pd.DataFrame()

        columns = [row[1] for row in table_info]

        try:
            if "player_id" in columns and "value" in columns:
                where_clauses = ["player_id = ?"]
                params: list[object] = [player_id]

                if scenario_name is not None and "scenario" in columns:
                    where_clauses.append("scenario = ?")
                    params.append(scenario_name)

                if policy_name is not None and "policy" in columns:
                    where_clauses.append("policy = ?")
                    params.append(policy_name)

                query = f"""
                    SELECT period, value
                    FROM {table_name}
                    WHERE {' AND '.join(where_clauses)}
                    ORDER BY period
                """
                result = pd.read_sql_query(query, conn, params=params)
            else:
                player_col = str(player_id)
                if player_col not in columns:
                    return pd.DataFrame()

                if "period" in columns:
                    query = f"""
                        SELECT period, "{player_col}" AS value
                        FROM {table_name}
                        ORDER BY period
                    """
                else:
                    query = f"""
                        SELECT "{player_col}" AS value
                        FROM {table_name}
                    """

                result = pd.read_sql_query(query, conn)
        except (pd.errors.DatabaseError, sqlite3.OperationalError):
            return pd.DataFrame()

    if result.empty:
        return result

    if "period" in result.columns:
        result["hour"] = result["period"] / 4.0

    return result


def load_attribute(table_name: str, player_id: int, attribute_name: str):
    if not _is_safe_identifier(table_name):
        raise ValueError(f"Invalid table name: {table_name}")

    if not _is_safe_identifier(attribute_name):
        raise ValueError(f"Invalid attribute name: {attribute_name}")

    with sqlite3.connect(Config.SQLITE_PATH) as conn:
        try:
            row = conn.execute(
                f"SELECT {attribute_name} FROM {table_name} WHERE player_id = ? LIMIT 1",
                (player_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None

    if row is None:
        return None

    return row[0]


def load_avg_profile(
    policy_name: str,
    scenario_name: str,
    table_name: str,
) -> pd.DataFrame:
    if not _is_safe_identifier(table_name):
        raise ValueError(f"Invalid table name: {table_name}")
    
    with sqlite3.connect(Config.SQLITE_PATH) as conn:
        try:
            result = pd.read_sql_query(
                f"""
                SELECT period, AVG(value) AS value
                FROM {table_name}
                WHERE policy = ? AND scenario = ?
                GROUP BY period
                ORDER BY period
                """,
                conn,
                params=(policy_name, scenario_name),
            )
        except (pd.errors.DatabaseError, sqlite3.OperationalError):
            return pd.DataFrame()

    if result.empty:
        return result

    result["policy"] = policy_name
    result["hour"] = result["period"] / 4.0
    return result


def load_policies() -> list[str]:
    with sqlite3.connect(Config.SQLITE_PATH) as conn:
        rows = conn.execute(
            "SELECT DISTINCT policy FROM results ORDER BY policy"
        ).fetchall()
    return [row[0] for row in rows]


def load_scenarios() -> list[str]:
    with sqlite3.connect(Config.SQLITE_PATH) as conn:
        rows = conn.execute(
            "SELECT DISTINCT scenario FROM results ORDER BY scenario"
        ).fetchall()
    return [row[0] for row in rows]


def load_household_ids() -> list[int]:
    with sqlite3.connect(Config.SQLITE_PATH) as conn:
        rows = conn.execute(
            "SELECT DISTINCT player_id FROM results ORDER BY player_id"
        ).fetchall()
    return [row[0] for row in rows]


def load_household_kpi_result(
    player_id: int,
    scenario_name: str,
    policy_name: str,
) -> pd.DataFrame:
    with sqlite3.connect(Config.SQLITE_PATH) as conn:
        try:
            result = pd.read_sql_query(
                """
                SELECT
                    policy,
                    scenario,
                    player_id,
                    total_cost,
                    target_met_bess,
                    target_met_ev1,
                    target_met_ev2,
                    soc_at_deadline_bess,
                    soc_at_deadline_ev1,
                    soc_at_deadline_ev2
                FROM results
                WHERE player_id = ? AND scenario = ? AND policy = ?
                ORDER BY rowid DESC
                LIMIT 1
                """,
                conn,
                params=(player_id, scenario_name, policy_name),
            )
        except (pd.errors.DatabaseError, sqlite3.OperationalError):
            return pd.DataFrame()

    return result


def fetch_timeseries(sqlite_cursor, player_id, measurement):
    data = sqlite_cursor.execute(
        f'''
        SELECT "{player_id}" as value
        FROM {measurement}
        ''',
    ).fetchall()

    return [row[0] for row in data]


def fetch_multiple_timeseries(sqlite_cursor, player_id, measurements):
    timeseries_data = {measurement: [] for measurement in measurements}
    for measurement in measurements:
        data = fetch_timeseries(sqlite_cursor, player_id, measurement)
        timeseries_data[measurement] = data

    return timeseries_data


if __name__ == "__main__":
    # Example usage
    player_id = 12
    measurement = 'base_load'
    result = load_household_kpi_result(player_id, "default_scenario", "no_control")
    print(result)