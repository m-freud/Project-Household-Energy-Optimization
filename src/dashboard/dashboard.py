from pathlib import Path
import sys

# Ensure absolute imports like `from src...` work even when launched from a subdirectory.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
	sys.path.insert(0, str(repo_root))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
#~

from src.sqlite_connection import (
	load_household_ids_from_results as db_load_household_ids,
	load_policies as db_load_policies,
	load_source_household_ids,
)
from src.simulation.scenarios.scenario import get_scenario_names, get_scenario_summary_rows

from src.dashboard.general_performance.general_performance import render_general_performance
from src.dashboard.single_performance.single_performance import render_single_performance
from src.dashboard.prediction_explorer import render_prediction_explorer


@st.cache_data(show_spinner=False)
def load_policies() -> list[str]:
	return db_load_policies()


@st.cache_data(show_spinner=False)
def load_scenarios() -> list[str]:
	return get_scenario_names()


@st.cache_data(show_spinner=False)
def load_scenario_summary() -> pd.DataFrame:
	return pd.DataFrame(get_scenario_summary_rows())


@st.cache_data(show_spinner=False)
def load_household_ids() -> list[int]:
	result_ids = db_load_household_ids()
	source_ids = load_source_household_ids()
	merged = sorted(set(result_ids + source_ids))
	return merged


def main():
	st.set_page_config(page_title="Household Energy Optimization Dashboard", layout="wide")
	st.title("Household Energy Optimization Dashboard")
	
	policies = load_policies()
	scenarios = load_scenarios()
	scenario_summary = load_scenario_summary()
	household_ids = load_household_ids()

	with st.expander("Scenario Grid", expanded=False):
		st.caption("Active benchmark scenarios with dashboard labels and descriptions")
		st.dataframe(scenario_summary, width="stretch")

	if not scenarios or not household_ids:
		st.warning("No household/source data found. Run ingestion first.")
		return

	if policies:
		render_general_performance(
			policies,
			scenarios
		)

		st.divider()

		render_single_performance(
			policies=policies,
			scenarios=scenarios,
			household_ids=household_ids,
		)

		st.divider()
	else:
		st.info("No simulation results yet. Prediction Explorer is still available below.")
		st.divider()

	render_prediction_explorer(
		household_ids=household_ids,
	)

	st.divider()
	

if __name__ == "__main__":
	main()
