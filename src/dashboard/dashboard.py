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
	load_household_ids as db_load_household_ids,
	load_policies as db_load_policies,
	load_scenarios as db_load_scenarios,
)
from src.dashboard.overview.overview import render_overview_section
from src.dashboard.single_household_view.single_household_view import plot_household_overview
from src.dashboard.single_household_view.more_tables import (
	build_debug_table as db_build_debug_table,
	build_kpi_table as db_build_kpi_table,
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

	st.subheader("Single Household Overview")
	if not selected_single_policies:
		st.info("Select at least one policy.")
		return

	figure = plot_household_overview(
		player_id=selected_player_id,
		scenario_name=selected_single_scenario,
		policy_names=selected_single_policies,
	)
	kpi_df = db_build_kpi_table(
		player_id=selected_player_id,
		scenario_name=selected_single_scenario,
		policy_names=selected_single_policies,
	)
	debug_df = db_build_debug_table(
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

