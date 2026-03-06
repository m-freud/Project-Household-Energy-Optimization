import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.sqlite_connection import load_series
from src.simulation.scenarios.scenario import get_scenario_value


def _shade_ev_location_background(ax, at_home_df: pd.DataFrame, at_station_df: pd.DataFrame) -> None:
	if at_home_df.empty or at_station_df.empty:
		return

	merged = at_home_df[["hour", "value"]].rename(columns={"value": "at_home"}).merge(
		at_station_df[["hour", "value"]].rename(columns={"value": "at_station"}),
		on="hour",
		how="inner",
	)

	if merged.empty:
		return

	hours = merged["hour"].tolist()
	at_home_values = merged["at_home"].tolist()
	at_station_values = merged["at_station"].tolist()

	for idx, start_hour in enumerate(hours):
		if idx < len(hours) - 1:
			end_hour = hours[idx + 1]
		else:
			end_hour = start_hour + 0.25

		if at_station_values[idx] == 1:
			color = "lightblue"
		elif at_home_values[idx] == 1:
			color = "lightgreen"
		else:
			color = "lightgrey"

		ax.axvspan(start_hour, end_hour, color=color, alpha=0.22, linewidth=0)


def plot_single_household_view(
	player_id: int,
	scenario_name: str,
	policy_names: list[str],
):
	pv_gen_df = load_series("pv_gen", player_id)
	ev1_at_home_df = load_series("ev1_at_home", player_id)
	ev1_at_station_df = load_series("ev1_at_charging_station", player_id)
	ev2_at_home_df = load_series("ev2_at_home", player_id)
	ev2_at_station_df = load_series("ev2_at_charging_station", player_id)

	fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)
	kpi_rows = []
	debug_rows = []

	policy_cmap = plt.get_cmap("tab10")
	policy_colors = {
		policy_name: policy_cmap(i % 10)
		for i, policy_name in enumerate(policy_names)
	}

	# PV
	pv_ax = axes[1, 0]
	pv_ax.set_title("PV")
	pv_ax.set_ylabel("Power (kW)")
	pv_ax.set_xlabel("Hour")
	if pv_gen_df.empty:
		pv_ax.text(0.5, 0.5, "No PV data", transform=pv_ax.transAxes, ha="center", va="center")
	else:
		pv_ax.plot(pv_gen_df["hour"], pv_gen_df["value"], label="PV Gen", color="tab:orange", linewidth=2)
		pv_ax.fill_between(pv_gen_df["hour"], pv_gen_df["value"], color="tab:orange", alpha=0.2)
		pv_ax.legend(loc="upper left")

	# BESS
	bess_ax = axes[0, 0]
	bess_ax.set_title("BESS")
	bess_ax.set_ylabel("SOC (kWh)")
	bess_has_data = False
	target_soc_bess = get_scenario_value(scenario_name, "bess", player_id, "target_soc")
	bess_deadline = get_scenario_value(scenario_name, "bess", player_id, "deadline")
	if target_soc_bess is not None:
		bess_ax.axhline(
			y=target_soc_bess,
			color="tab:red",
			linestyle="--",
			linewidth=1.5,
		)

	for policy_name in policy_names:
		color = policy_colors[policy_name]
		bess_soc_df = load_series("bess_soc", player_id, scenario_name, policy_name)
		net_load_df = load_series("net_load", player_id, scenario_name, policy_name)
		net_cost_df = load_series("net_cost", player_id, scenario_name, policy_name)
		total_cost_df = load_series("total_cost", player_id, scenario_name, policy_name)

		if not bess_soc_df.empty:
			bess_has_data = True
			bess_ax.plot(bess_soc_df["hour"], bess_soc_df["value"], color=color, linewidth=2)

		if not net_load_df.empty:
			peak_load_idx = net_load_df["value"].idxmax()
			peak_load_hour = float(net_load_df.loc[peak_load_idx, "hour"])
			peak_load_val = float(net_load_df.loc[peak_load_idx, "value"])
			imported_energy_kwh = float(net_load_df["value"].clip(lower=0).sum() * 0.25)
			exported_energy_kwh = float((-net_load_df["value"].clip(upper=0)).sum() * 0.25)
		else:
			peak_load_hour = None
			peak_load_val = None
			imported_energy_kwh = 0.0
			exported_energy_kwh = 0.0

		if not net_cost_df.empty:
			peak_cost_idx = net_cost_df["value"].idxmax()
			peak_cost_hour = float(net_cost_df.loc[peak_cost_idx, "hour"])
		else:
			peak_cost_hour = None

		if not total_cost_df.empty:
			total_cost_value = float(total_cost_df["value"].iloc[-1])
		else:
			total_cost_value = float(net_cost_df["value"].sum() * 0.25) if not net_cost_df.empty else 0.0

		target_status = {}
		for device_name in ["bess", "ev1", "ev2"]:
			target_soc_kwh = get_scenario_value(scenario_name, device_name, player_id, "target_soc")
			deadline_period = get_scenario_value(scenario_name, device_name, player_id, "deadline")
			soc_df = load_series(f"{device_name}_soc", player_id, scenario_name, policy_name)

			deadline_hour = None if deadline_period is None else float(deadline_period) / 4.0
			soc_at_deadline = None
			met_target = None
	
			if deadline_period is not None and not soc_df.empty:
				print("A")
				at_deadline_df = soc_df[soc_df["period"] == int(deadline_period)]
				if not at_deadline_df.empty:
					soc_at_deadline = float(at_deadline_df["value"].iloc[0])
					if target_soc_kwh is not None:
						met_target = soc_at_deadline >= float(target_soc_kwh)

			target_status[device_name] = met_target
			print(target_status)
			print(f"Device: {device_name.upper()}, Deadline Hour: {deadline_hour}, SOC at Deadline: {soc_at_deadline}, Target SOC: {target_soc_kwh}, Met Target: {met_target}")

			debug_rows.append(
				{
					"policy": policy_name,
					"device": device_name.upper(),
					"deadline_hour": deadline_hour,
					"soc_at_deadline_kwh": soc_at_deadline,
					"target_soc_kwh": None if target_soc_kwh is None else float(target_soc_kwh),
					"deadline_missed": None if met_target is None else (not met_target),
					"peak_load_hour": peak_load_hour,
					"peak_cost_hour": peak_cost_hour,
				}
			)

		kpi_rows.append(
			{
				"policy": policy_name,
				"total_cost": total_cost_value,
				"peak_load": peak_load_val,
				"imported_energy_kwh": imported_energy_kwh,
				"exported_energy_kwh": exported_energy_kwh,
				"bess_target_met": target_status["bess"],
				"ev1_target_met": target_status["ev1"],
				"ev2_target_met": target_status["ev2"],
			}
		)

	if not bess_has_data:
		bess_ax.text(0.5, 0.5, "No BESS SOC data", transform=bess_ax.transAxes, ha="center", va="center")

	if bess_deadline is not None:
		bess_deadline_hour = float(bess_deadline) / 4.0
		bess_ax.axvline(x=bess_deadline_hour, color="darkblue", linewidth=1.8)
		bess_missed = any(
			row["device"] == "BESS" and row["deadline_missed"] is True
			for row in debug_rows
		)
		if bess_missed:
			bess_ax.axvline(x=bess_deadline_hour + 0.03, color="red", linewidth=1.8)

	# EV1
	ev1_ax = axes[0, 1]
	ev1_ax.set_title("EV1")
	ev1_ax.set_ylabel("SOC (kWh)")
	_shade_ev_location_background(ev1_ax, ev1_at_home_df, ev1_at_station_df)
	ev1_has_data = False
	target_soc_ev1 = get_scenario_value(scenario_name, "ev1", player_id, "target_soc")
	ev1_deadline = get_scenario_value(scenario_name, "ev1", player_id, "deadline")
	if target_soc_ev1 is not None:
		ev1_ax.axhline(
			y=target_soc_ev1,
			color="tab:red",
			linestyle="--",
			linewidth=1.5,
		)

	for policy_name in policy_names:
		ev1_soc_df = load_series("ev1_soc", player_id, scenario_name, policy_name)
		if not ev1_soc_df.empty:
			ev1_has_data = True
			ev1_ax.plot(ev1_soc_df["hour"], ev1_soc_df["value"], color=policy_colors[policy_name], linewidth=2)

	if not ev1_has_data:
		ev1_ax.text(0.5, 0.5, "No EV1 SOC data", transform=ev1_ax.transAxes, ha="center", va="center")

	if ev1_deadline is not None:
		ev1_deadline_hour = float(ev1_deadline) / 4.0
		ev1_ax.axvline(x=ev1_deadline_hour, color="darkblue", linewidth=1.8)
		ev1_missed = any(
			row["device"] == "EV1" and row["deadline_missed"] is True
			for row in debug_rows
		)
		if ev1_missed:
			ev1_ax.axvline(x=ev1_deadline_hour + 0.03, color="red", linewidth=1.8)

	# EV2
	ev2_ax = axes[0, 2]
	ev2_ax.set_title("EV2")
	ev2_ax.set_ylabel("SOC (kWh)")
	_shade_ev_location_background(ev2_ax, ev2_at_home_df, ev2_at_station_df)
	ev2_has_data = False
	target_soc_ev2 = get_scenario_value(scenario_name, "ev2", player_id, "target_soc")
	ev2_deadline = get_scenario_value(scenario_name, "ev2", player_id, "deadline")
	if target_soc_ev2 is not None:
		ev2_ax.axhline(
			y=target_soc_ev2,
			color="tab:red",
			linestyle="--",
			linewidth=1.5,
		)

	for policy_name in policy_names:
		ev2_soc_df = load_series("ev2_soc", player_id, scenario_name, policy_name)
		if not ev2_soc_df.empty:
			ev2_has_data = True
			ev2_ax.plot(ev2_soc_df["hour"], ev2_soc_df["value"], color=policy_colors[policy_name], linewidth=2)

	if not ev2_has_data:
		ev2_ax.text(0.5, 0.5, "No EV2 SOC data", transform=ev2_ax.transAxes, ha="center", va="center")

	if ev2_deadline is not None:
		ev2_deadline_hour = float(ev2_deadline) / 4.0
		ev2_ax.axvline(x=ev2_deadline_hour, color="darkblue", linewidth=1.8)
		ev2_missed = any(
			row["device"] == "EV2" and row["deadline_missed"] is True
			for row in debug_rows
		)
		if ev2_missed:
			ev2_ax.axvline(x=ev2_deadline_hour + 0.03, color="red", linewidth=1.8)

	# Net Load
	net_load_ax = axes[1, 1]
	net_load_ax.set_title("Net Load")
	net_load_ax.set_ylabel("Power (kW)")
	net_load_ax.set_xlabel("Hour")
	net_load_has_data = False
	for policy_name in policy_names:
		net_load_df = load_series("net_load", player_id, scenario_name, policy_name)
		if not net_load_df.empty:
			net_load_has_data = True
			net_load_ax.plot(net_load_df["hour"], net_load_df["value"], color=policy_colors[policy_name], linewidth=2)
	if not net_load_has_data:
		net_load_ax.text(0.5, 0.5, "No net load data", transform=net_load_ax.transAxes, ha="center", va="center")
	net_load_ax.axhline(y=0.0, color="black", linewidth=1, alpha=0.5)

	# Net Cost
	net_cost_ax = axes[1, 2]
	net_cost_ax.set_title("Net Cost")
	net_cost_ax.set_ylabel("Cost")
	net_cost_ax.set_xlabel("Hour")
	net_cost_has_data = False
	for policy_name in policy_names:
		net_cost_df = load_series("net_cost", player_id, scenario_name, policy_name)
		if not net_cost_df.empty:
			net_cost_has_data = True
			net_cost_ax.plot(net_cost_df["hour"], net_cost_df["value"], color=policy_colors[policy_name], linewidth=2)
	if not net_cost_has_data:
		net_cost_ax.text(0.5, 0.5, "No net cost data", transform=net_cost_ax.transAxes, ha="center", va="center")
	net_cost_ax.axhline(y=0.0, color="black", linewidth=1, alpha=0.5)

	policy_handles = [
		Line2D([0], [0], color=policy_colors[policy_name], linewidth=2, label=policy_name)
		for policy_name in policy_names
	]
	style_handles = [
		Line2D([0], [0], color="tab:red", linestyle="--", linewidth=1.5, label="Target SOC"),
		Line2D([0], [0], color="darkblue", linewidth=1.8, label="Deadline"),
		Line2D([0], [0], color="red", linewidth=1.8, label="Missed Deadline"),
	]
	fig.legend(
		handles=policy_handles + style_handles,
		loc="upper center",
		ncol=min(len(policy_handles) + len(style_handles), 8),
		bbox_to_anchor=(0.5, 1.02),
	)

	for ax in axes.flatten():
		ax.grid(alpha=0.25)

	fig.tight_layout(rect=[0, 0, 1, 0.97])

	kpi_df = pd.DataFrame(kpi_rows)
	if not kpi_df.empty:
		kpi_df = kpi_df.drop_duplicates(subset=["policy"]).reset_index(drop=True)

	debug_df = pd.DataFrame(debug_rows)
	return fig, kpi_df, debug_df


if __name__ == "__main__":
	player_id = 1
	scenario_name = "default_scenario"
	policy_names = ["no_control"]

	fig, kpi_df, debug_df = plot_single_household_view(player_id, scenario_name, policy_names)
	print(kpi_df)
	print(debug_df)
	fig.show()