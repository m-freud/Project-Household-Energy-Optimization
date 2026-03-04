from pathlib import Path
import sqlite3
import sys

import pandas as pd
import streamlit as st


repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.config import Config


DEFAULT_MEASUREMENTS = [
    "base_load",
    "pv_gen",
    "net_load",
    "bess_soc",
    "ev1_soc",
    "ev2_soc",
    "total_cost",
]


def get_connection():
    return sqlite3.connect(Config.SQLITE_PATH)


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    result = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    ).fetchone()
    return result is not None


def get_measurement_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    tables = [r[0] for r in rows]

    available = []
    for table in tables:
        if table in {"results", "sqlite_sequence"}:
            continue

        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        col_names = {c[1] for c in cols}
        if {"player_id", "policy", "scenario", "period", "value"}.issubset(col_names):
            available.append(table)

    return available


def get_filter_options(conn: sqlite3.Connection, measurement: str) -> tuple[list[int], list[str], list[str], int]:
    players = conn.execute(
        f"SELECT DISTINCT player_id FROM {measurement} ORDER BY player_id"
    ).fetchall()
    policies = conn.execute(
        f"SELECT DISTINCT policy FROM {measurement} ORDER BY policy"
    ).fetchall()
    scenarios = conn.execute(
        f"SELECT DISTINCT scenario FROM {measurement} ORDER BY scenario"
    ).fetchall()
    max_period = conn.execute(
        f"SELECT COALESCE(MAX(period), 0) FROM {measurement}"
    ).fetchone()[0]

    num_days = int(max_period // 96) + 1
    return [p[0] for p in players], [p[0] for p in policies], [s[0] for s in scenarios], num_days


def fetch_measurement_series(
    conn: sqlite3.Connection,
    measurement: str,
    player_id: int,
    policy: str,
    scenario: str,
    day: int,
) -> pd.DataFrame:
    day_start = day * 96
    day_end = day_start + 95

    query = f"""
        SELECT period, value
        FROM {measurement}
        WHERE player_id = ?
          AND policy = ?
          AND scenario = ?
          AND period BETWEEN ? AND ?
        ORDER BY period
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(player_id, policy, scenario, day_start, day_end),
    )

    if df.empty:
        return df

    df["step_of_day"] = df["period"] - day_start
    df["hour"] = df["step_of_day"] / 4.0
    df["measurement"] = measurement
    return df


def fetch_day_summary(
    conn: sqlite3.Connection,
    player_id: int,
    policy: str,
    scenario: str,
) -> pd.DataFrame:
    if not table_exists(conn, "results"):
        return pd.DataFrame()

    df = pd.read_sql_query(
        """
        SELECT player_id, policy, scenario, total_cost, total_consumption
        FROM results
        WHERE player_id = ? AND policy = ? AND scenario = ?
        """,
        conn,
        params=(player_id, policy, scenario),
    )
    return df


def main():
    st.set_page_config(page_title="Household Energy Dashboard", layout="wide")
    st.title("Household Energy Dashboard")

    conn = get_connection()
    measurement_tables = get_measurement_tables(conn)

    if not measurement_tables:
        st.error("No simulation measurement tables found. Run the simulation first.")
        conn.close()
        return

    base_measurement = next(
        (m for m in ["net_load", "total_cost", "base_load"] if m in measurement_tables),
        measurement_tables[0],
    )

    players, policies, scenarios, num_days = get_filter_options(conn, base_measurement)
    if not players or not policies or not scenarios:
        st.warning("No selectable player/policy/scenario values found.")
        conn.close()
        return

    st.sidebar.header("Filters")
    player_id = st.sidebar.selectbox("Player", players, index=0)
    policy = st.sidebar.selectbox("Policy", policies, index=0)
    scenario = st.sidebar.selectbox("Scenario", scenarios, index=0)
    day = st.sidebar.slider("Day", min_value=0, max_value=max(0, num_days - 1), value=0)

    default_selected = [m for m in DEFAULT_MEASUREMENTS if m in measurement_tables]
    selected_measurements = st.sidebar.multiselect(
        "Measurements",
        options=measurement_tables,
        default=default_selected[:4] if default_selected else measurement_tables[:3],
    )

    view = st.radio("View", ["Timeseries", "Summary"], horizontal=True)

    if view == "Timeseries":
        if not selected_measurements:
            st.info("Select at least one measurement.")
        else:
            chart_data = []
            for measurement in selected_measurements:
                series_df = fetch_measurement_series(conn, measurement, player_id, policy, scenario, day)
                if not series_df.empty:
                    chart_data.append(series_df)

            if not chart_data:
                st.warning("No data for this filter combination.")
            else:
                all_data = pd.concat(chart_data, ignore_index=True)
                st.line_chart(all_data, x="hour", y="value", color="measurement")
                st.dataframe(
                    all_data[["measurement", "period", "hour", "value"]],
                    use_container_width=True,
                    hide_index=True,
                )

    if view == "Summary":
        summary_df = fetch_day_summary(conn, player_id, policy, scenario)
        if summary_df.empty:
            st.info("No summary rows in results for this selection.")
        else:
            row = summary_df.iloc[0]
            col1, col2 = st.columns(2)
            col1.metric("Total Cost", f"{row['total_cost']:.2f}")
            col2.metric("Total Consumption", f"{row['total_consumption']:.2f}")
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    conn.close()


if __name__ == "__main__":
    main()
