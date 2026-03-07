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


def build_kpi_table(
	player_id: int,
	scenario_name: str,
	policy_names: list[str],
) -> pd.DataFrame:
	kpi_rows = []

	for policy_name in policy_names:
		result_df = load_household_result(player_id, scenario_name, policy_name)
		net_load_df = load_series("net_load", player_id, scenario_name, policy_name)
		net_cost_df = load_series("net_cost", player_id, scenario_name, policy_name)
		total_cost_df = load_series("total_cost", player_id, scenario_name, policy_name)

		if not net_load_df.empty:
			peak_load_idx = net_load_df["value"].idxmax()
			peak_load_val = float(net_load_df.loc[peak_load_idx, "value"])
			imported_energy_kwh = float(net_load_df["value"].clip(lower=0).sum() * 0.25)
			exported_energy_kwh = float((-net_load_df["value"].clip(upper=0)).sum() * 0.25)
		else:
			peak_load_val = None
			imported_energy_kwh = 0.0
			exported_energy_kwh = 0.0

		if not total_cost_df.empty:
			total_cost_value = float(total_cost_df["value"].iloc[-1])
		else:
			total_cost_value = float(net_cost_df["value"].sum() * 0.25) if not net_cost_df.empty else 0.0

		target_met_bess = None
		target_met_ev1 = None
		target_met_ev2 = None
		if not result_df.empty:
			result_row = result_df.iloc[0]
			target_met_bess = _to_optional_bool(result_row.get("target_met_bess"))
			target_met_ev1 = _to_optional_bool(result_row.get("target_met_ev1"))
			target_met_ev2 = _to_optional_bool(result_row.get("target_met_ev2"))
			if pd.notna(result_row.get("total_cost")):
				total_cost_value = float(result_row["total_cost"])

		kpi_rows.append(
			{
				"policy": policy_name,
				"total_cost": total_cost_value,
				"peak_load": peak_load_val,
				"imported_energy_kwh": imported_energy_kwh,
				"exported_energy_kwh": exported_energy_kwh,
				"target_met_bess": target_met_bess,
				"target_met_ev1": target_met_ev1,
				"target_met_ev2": target_met_ev2,
			}
		)

	kpi_df = pd.DataFrame(kpi_rows)
	if not kpi_df.empty:
		kpi_df = kpi_df.drop_duplicates(subset=["policy"]).reset_index(drop=True)
	return kpi_df


def render_kpi_strip(
	player_id: int,
	scenario_name: str,
	policy_names: list[str],
) -> None:
	kpi_df = build_kpi_table(player_id, scenario_name, policy_names)

	st.subheader("KPI Strip")

	if kpi_df.empty:
		st.info("No KPI data available for selected inputs.")
		return

	kpi_display = kpi_df.copy()
	
	for col in ["total_cost", "peak_load", "imported_energy_kwh", "exported_energy_kwh"]:
		kpi_display[col] = kpi_display[col].map(lambda value: round(value, 2) if pd.notna(value) else None)

	st.dataframe(kpi_display, use_container_width=True)
