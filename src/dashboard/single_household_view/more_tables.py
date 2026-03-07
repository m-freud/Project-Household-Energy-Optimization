import pandas as pd

from src.sqlite_connection import load_household_result, load_household_result, load_series


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
