import streamlit as st

import pandas as pd
# paste this to enable src. imports
from pathlib import Path
import sys
# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.sqlite_connection import load_household_result, load_series


def _to_optional_bool(value):
	if pd.isna(value):
		return None
	return bool(int(value))


def build_debug_table(
	player_id: int,
	scenario_name: str,
	policy_names: list[str],
) -> pd.DataFrame:
	debug_rows = []

	for policy_name in policy_names:
		result_df = load_household_result(player_id, scenario_name, policy_name)
		net_load_df = load_series("net_load", player_id, scenario_name, policy_name)
		net_cost_df = load_series("net_cost", player_id, scenario_name, policy_name)

		if not net_load_df.empty:
			peak_load_idx = net_load_df["value"].idxmax()
			peak_load_hour = float(net_load_df.loc[peak_load_idx, "hour"])
		else:
			peak_load_hour = None

		if not net_cost_df.empty:
			peak_cost_idx = net_cost_df["value"].idxmax()
			peak_cost_hour = float(net_cost_df.loc[peak_cost_idx, "hour"])
		else:
			peak_cost_hour = None

		target_met_bess = None
		target_met_ev1 = None
		target_met_ev2 = None
		soc_at_deadline_bess = None
		soc_at_deadline_ev1 = None
		soc_at_deadline_ev2 = None

		if not result_df.empty:
			result_row = result_df.iloc[0]
			target_met_bess = _to_optional_bool(result_row.get("target_met_bess"))
			target_met_ev1 = _to_optional_bool(result_row.get("target_met_ev1"))
			target_met_ev2 = _to_optional_bool(result_row.get("target_met_ev2"))
			soc_at_deadline_bess = result_row.get("soc_at_deadline_bess")
			soc_at_deadline_ev1 = result_row.get("soc_at_deadline_ev1")
			soc_at_deadline_ev2 = result_row.get("soc_at_deadline_ev2")

		device_map = {
			"BESS": (target_met_bess, soc_at_deadline_bess),
			"EV1": (target_met_ev1, soc_at_deadline_ev1),
			"EV2": (target_met_ev2, soc_at_deadline_ev2),
		}

		for device_name, (met_target, soc_at_deadline_value) in device_map.items():
			debug_rows.append(
				{
					"policy": policy_name,
					"device": device_name,
					"deadline_hour": None,
					"soc_at_deadline_kwh": None if pd.isna(soc_at_deadline_value) else float(soc_at_deadline_value),
					"target_soc_kwh": None,
					"deadline_missed": None if met_target is None else (not met_target),
					"peak_load_hour": peak_load_hour,
					"peak_cost_hour": peak_cost_hour,
				}
			)

	return pd.DataFrame(debug_rows)



def render_debug_table(
	player_id: int,
	scenario_name: str,
	policy_names: list[str],
) -> None:
	debug_df = build_debug_table(player_id, scenario_name, policy_names)

	st.subheader("Debug Table")
	if debug_df.empty:
		st.info("No debug timestamps available for selected inputs.")
	else:
		st.dataframe(debug_df, use_container_width=True)
		
