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

from src.dashboard_2.general_performance.general_performance import render_general_performance
from src.dashboard_2.single_performance.single_performance import render_single_performance


@st.cache_data(show_spinner=False)
def load_policies() -> list[str]:
	return db_load_policies()


@st.cache_data(show_spinner=False)
def load_scenarios() -> list[str]:
	return db_load_scenarios()


@st.cache_data(show_spinner=False)
def load_household_ids() -> list[int]:
	return db_load_household_ids()


def main():
	st.set_page_config(page_title="Household Energy Optimization Dashboard", layout="wide")
	st.title("Household Energy Optimization Dashboard")
	
	policies = load_policies()
	scenarios = load_scenarios()
	household_ids = load_household_ids()

	if not policies or not scenarios or not household_ids:
		st.warning("No rows found in total_cost. Run a simulation first.")
		return
	
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
	

if __name__ == "__main__":
	main()
