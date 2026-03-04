import sqlite3

import pandas as pd
import streamlit as st
# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.config import Config


@st.cache_data(show_spinner=False)
def load_policy_averages() -> pd.DataFrame:
	with sqlite3.connect(Config.SQLITE_PATH) as conn:
		return pd.read_sql_query(
			"""
			SELECT policy, AVG(total_cost) AS avg_total_cost
			FROM results
			GROUP BY policy
			ORDER BY avg_total_cost ASC
			""",
			conn,
		)


@st.cache_data(show_spinner=False)
def load_total_cost_avg(policy_name: str, scenario_name: str) -> pd.DataFrame:
	with sqlite3.connect(Config.SQLITE_PATH) as conn:
		result = pd.read_sql_query(
			"""
			SELECT period, AVG(value) AS avg_total_cost
			FROM total_cost
			WHERE policy = ? AND scenario = ?
			GROUP BY period
			ORDER BY period
			""",
			conn,
			params=(policy_name, scenario_name),
		)

	if result.empty:
		return result

	result["policy"] = policy_name
	result["hour"] = result["period"] / 4.0
	return result


@st.cache_data(show_spinner=False)
def load_policies() -> list[str]:
	with sqlite3.connect(Config.SQLITE_PATH) as conn:
		rows = conn.execute(
			"SELECT DISTINCT policy FROM total_cost ORDER BY policy"
		).fetchall()
	return [row[0] for row in rows]


@st.cache_data(show_spinner=False)
def load_scenarios() -> list[str]:
	with sqlite3.connect(Config.SQLITE_PATH) as conn:
		rows = conn.execute(
			"SELECT DISTINCT scenario FROM total_cost ORDER BY scenario"
		).fetchall()
	return [row[0] for row in rows]


def main():
	st.set_page_config(page_title="Policy Cost Comparison", layout="wide")
	st.title("Policy Comparison")
	st.caption("Average total cost across all households")

	policies = load_policies()
	scenarios = load_scenarios()

	if not policies or not scenarios:
		st.warning("No rows found in total_cost. Run a simulation first.")
		return

	left_col, right_col = st.columns([1, 3], gap="large")

	with left_col:
		st.subheader("Policies")
		selected_scenario = st.selectbox("Scenario", options=scenarios, index=0)
		selected_policies = []
		for policy_name in policies:
			if st.checkbox(policy_name, value=True):
				selected_policies.append(policy_name)

	series_frames = []
	for policy_name in selected_policies:
		series_df = load_total_cost_avg(policy_name, selected_scenario)
		if not series_df.empty:
			series_frames.append(series_df)

	with right_col:
		st.subheader("Average Total Cost Over Time")
		if not series_frames:
			st.info("Select at least one policy.")
		else:
			chart_df = pd.concat(series_frames, ignore_index=True)
			pivot_df = chart_df.pivot(index="hour", columns="policy", values="avg_total_cost")
			st.line_chart(pivot_df)


if __name__ == "__main__":
	main()

