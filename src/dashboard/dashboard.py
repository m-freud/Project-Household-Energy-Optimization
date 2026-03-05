import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.sqlite_connection import (
	load_avg_profile as db_load_avg_profile,
	load_household_ids as db_load_household_ids,
	load_policies as db_load_policies,
	load_scenarios as db_load_scenarios,
)
from src.dashboard.single_household_view import plot_single_household_view


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


@st.cache_data(show_spinner=False)
def load_policies() -> list[str]:
	return db_load_policies()


@st.cache_data(show_spinner=False)
def load_scenarios() -> list[str]:
	return db_load_scenarios()


@st.cache_data(show_spinner=False)
def load_household_ids() -> list[int]:
	return db_load_household_ids()


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


def render_kpi_strip(kpi_df: pd.DataFrame) -> None:
	st.subheader("KPI Strip")
	if kpi_df.empty:
		st.info("No KPI data available for selected inputs.")
		return

	kpi_display = kpi_df.copy()
	for col in ["total_cost", "peak_load", "imported_energy_kwh", "exported_energy_kwh"]:
		kpi_display[col] = kpi_display[col].map(lambda value: round(value, 2) if pd.notna(value) else None)
	st.dataframe(kpi_display, use_container_width=True)


def render_debug_table(debug_df: pd.DataFrame) -> None:
	st.subheader("Debug Table")
	if debug_df.empty:
		st.info("No debug timestamps available for selected inputs.")
	else:
		st.dataframe(debug_df, use_container_width=True)


def render_single_view_section(
	policies: list[str],
	scenarios: list[str],
	household_ids: list[int],
) -> None:
	st.header("Single View")

	single_c1, single_c2, single_c3 = st.columns([1, 1, 1], gap="large")

	with single_c1:
		selected_player_id = st.selectbox("Household ID", options=household_ids, index=0)
	with single_c2:
		selected_single_scenario = st.selectbox(
			"Scenario",
			options=scenarios,
			index=0,
			key="single_view_scenario",
		)
	with single_c3:
		selected_single_policies = st.multiselect(
			"Policy",
			options=policies,
			default=policies[:2] if len(policies) >= 2 else policies,
			key="single_view_policy_multi",
		)

	st.subheader("Single Household")
	if not selected_single_policies:
		st.info("Select at least one policy.")
		return

	figure, kpi_df, debug_df = plot_single_household_view(
		player_id=selected_player_id,
		scenario_name=selected_single_scenario,
		policy_names=selected_single_policies,
	)

	st.pyplot(figure, use_container_width=True)
	plt.close(figure)

	render_kpi_strip(kpi_df)
	render_debug_table(debug_df)


def main():
	st.set_page_config(page_title="Household Energy Management", layout="wide")
	st.title("Household Energy Management - Analytics")
	st.caption("Average profiles across all households")

	policies = load_policies()
	scenarios = load_scenarios()
	household_ids = load_household_ids()

	if not policies or not scenarios or not household_ids:
		st.warning("No rows found in total_cost. Run a simulation first.")
		return

	render_overview_section(policies=policies, scenarios=scenarios)

	st.divider()
	render_single_view_section(
		policies=policies,
		scenarios=scenarios,
		household_ids=household_ids,
	)


if __name__ == "__main__":
	main()

