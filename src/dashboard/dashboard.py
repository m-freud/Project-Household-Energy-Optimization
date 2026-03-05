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


METRIC_CONFIG = {
	"avg cost": {
		"table": "net_cost",
		"value_col": "avg_cost",
		"title": "Average Cost Over Time",
	},
	"avg load": {
		"table": "net_load",
		"value_col": "avg_load",
		"title": "Average Load Over Time",
	},
	"avg total cost": {
		"table": "total_cost",
		"value_col": "avg_total_cost",
		"title": "Average Total Cost Over Time",
	},
	"avg total load": {
		"table": "total_consumption",
		"value_col": "avg_total_load",
		"title": "Average Total Load Over Time",
	},
}


@st.cache_data(show_spinner=False)
def load_avg_profile(policy_name: str, scenario_name: str, metric: str) -> pd.DataFrame:
	if metric not in METRIC_CONFIG:
		raise ValueError(f"Unsupported metric: {metric}")

	metric_config = METRIC_CONFIG[metric]
	table_name = metric_config["table"]
	value_col = metric_config["value_col"]

	with sqlite3.connect(Config.SQLITE_PATH) as conn:
		result = pd.read_sql_query(
			f"""
			SELECT period, AVG(value) AS {value_col}
			FROM {table_name}
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
			"SELECT DISTINCT policy FROM results ORDER BY policy"
		).fetchall()
	return [row[0] for row in rows]


@st.cache_data(show_spinner=False)
def load_scenarios() -> list[str]:
	with sqlite3.connect(Config.SQLITE_PATH) as conn:
		rows = conn.execute(
			"SELECT DISTINCT scenario FROM results ORDER BY scenario"
		).fetchall()
	return [row[0] for row in rows]


def main():
	st.set_page_config(page_title="Household Energy Management", layout="wide")
	st.title("Household Energy Management - Analytics")
	st.caption("Average profiles across all households")

	policies = load_policies()
	scenarios = load_scenarios()

	if not policies or not scenarios:
		st.warning("No rows found in total_cost. Run a simulation first.")
		return

	left_col, right_col = st.columns([1, 3], gap="large")

	with left_col:
		# TODO 
		st.subheader("Display Options")
		selected_scenario = st.selectbox("Scenario", options=scenarios, index=0)
		selected_metric = st.selectbox("Metric", options=list(METRIC_CONFIG.keys()), index=2)
		selected_policies = st.multiselect(
			"Policy",
			options=policies,
			default=policies,
		)

	series_frames = []
	for policy_name in selected_policies:
		series_df = load_avg_profile(policy_name, selected_scenario, selected_metric)
		if not series_df.empty:
			series_frames.append(series_df)

	with right_col:
		st.subheader(METRIC_CONFIG[selected_metric]["title"])
		if not series_frames:
			st.info("Select at least one policy.")
		else:
			chart_df = pd.concat(series_frames, ignore_index=True)
			pivot_df = chart_df.pivot(
				index="hour",
				columns="policy",
				values=METRIC_CONFIG[selected_metric]["value_col"],
			)
			st.line_chart(pivot_df)

	st.divider()


if __name__ == "__main__":
	main()

