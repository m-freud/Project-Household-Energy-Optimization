import pandas as pd
import streamlit as st

from src.sqlite_connection import load_avg_profile as db_load_avg_profile


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
	return db_load_avg_profile(
		policy_name=policy_name,
		scenario_name=scenario_name,
		table_name=metric_config["table"],
		value_col=metric_config["value_col"],
	)


def render_overview_section(policies: list[str], scenarios: list[str]) -> None:
	st.header("Overview")

	overview_c1, overview_c2, overview_c3 = st.columns([1, 1, 2], gap="large")

	with overview_c1:
		selected_scenario = st.selectbox("Scenario", options=scenarios, index=0)
	with overview_c2:
		selected_metric = st.selectbox("Metric", options=list(METRIC_CONFIG.keys()), index=2)
	with overview_c3:
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
