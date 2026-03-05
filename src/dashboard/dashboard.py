import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.sqlite_connection import load_series
from src.simulation.scenarios.scenario import get_scenario_value


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


@st.cache_data(show_spinner=False)
def load_household_ids() -> list[int]:
	with sqlite3.connect(Config.SQLITE_PATH) as conn:
		rows = conn.execute(
			"SELECT DISTINCT player_id FROM results ORDER BY player_id"
		).fetchall()
	return [row[0] for row in rows]

def shade_ev_location_background(ax, at_home_df: pd.DataFrame, at_station_df: pd.DataFrame) -> None:
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
	policy_name: str,
):
	bess_soc_df = load_series("bess_soc", player_id, scenario_name, policy_name)
	ev1_soc_df = load_series("ev1_soc", player_id, scenario_name, policy_name)
	ev2_soc_df = load_series("ev2_soc", player_id, scenario_name, policy_name)
	net_load_df = load_series("net_load", player_id, scenario_name, policy_name)
	net_cost_df = load_series("net_cost", player_id, scenario_name, policy_name)

	pv_gen_df = load_series("pv_gen", player_id)
	ev1_at_home_df = load_series("ev1_at_home", player_id)
	ev1_at_station_df = load_series("ev1_at_charging_station", player_id)
	ev2_at_home_df = load_series("ev2_at_home", player_id)
	ev2_at_station_df = load_series("ev2_at_charging_station", player_id)

	fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)

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
	if bess_soc_df.empty:
		bess_ax.text(0.5, 0.5, "No BESS SOC data", transform=bess_ax.transAxes, ha="center", va="center")
	else:
		bess_line = bess_ax.plot(bess_soc_df["hour"], bess_soc_df["value"], label="SOC", linewidth=2)
		target_soc_bess = get_scenario_value(scenario_name, "bess", player_id, "target_soc")
		if target_soc_bess is not None:
			target_line = bess_ax.axhline(
				y=target_soc_bess,
				color="tab:red",
				linestyle="--",
				linewidth=1.5,
				label="Target SOC",
			)
		else:
			target_line = None
		if target_line is not None:
			bess_ax.legend(handles=[bess_line[0], target_line], loc="upper left")
		else:
			bess_ax.legend(loc="upper left")

	# EV1
	ev1_ax = axes[0, 1]
	ev1_ax.set_title("EV1")
	ev1_ax.set_ylabel("SOC (kWh)")
	shade_ev_location_background(ev1_ax, ev1_at_home_df, ev1_at_station_df)
	if ev1_soc_df.empty:
		ev1_ax.text(0.5, 0.5, "No EV1 SOC data", transform=ev1_ax.transAxes, ha="center", va="center")
	else:
		ev1_ax.plot(ev1_soc_df["hour"], ev1_soc_df["value"], label="SOC", linewidth=2)
		target_soc_ev1 = get_scenario_value(scenario_name, "ev1", player_id, "target_soc")
		if target_soc_ev1 is not None:
			ev1_ax.axhline(
				y=target_soc_ev1,
				color="tab:red",
				linestyle="--",
				linewidth=1.5,
				label="Target SOC",
			)
		ev1_ax.legend(loc="upper left")

	# EV2
	ev2_ax = axes[0, 2]
	ev2_ax.set_title("EV2")
	ev2_ax.set_ylabel("SOC (kWh)")
	shade_ev_location_background(ev2_ax, ev2_at_home_df, ev2_at_station_df)
	if ev2_soc_df.empty:
		ev2_ax.text(0.5, 0.5, "No EV2 SOC data", transform=ev2_ax.transAxes, ha="center", va="center")
	else:
		ev2_ax.plot(ev2_soc_df["hour"], ev2_soc_df["value"], label="SOC", linewidth=2)
		target_soc_ev2 = get_scenario_value(scenario_name, "ev2", player_id, "target_soc")
		if target_soc_ev2 is not None:
			ev2_ax.axhline(
				y=target_soc_ev2,
				color="tab:red",
				linestyle="--",
				linewidth=1.5,
				label="Target SOC",
			)
		ev2_ax.legend(loc="upper left")

	# Net Load
	net_load_ax = axes[1, 1]
	net_load_ax.set_title("Net Load")
	net_load_ax.set_ylabel("Power (kW)")
	net_load_ax.set_xlabel("Hour")
	if net_load_df.empty:
		net_load_ax.text(0.5, 0.5, "No net load data", transform=net_load_ax.transAxes, ha="center", va="center")
	else:
		net_load_ax.plot(net_load_df["hour"], net_load_df["value"], label="Net Load", linewidth=2)
		net_load_ax.axhline(y=0.0, color="black", linewidth=1, alpha=0.5)
		net_load_ax.legend(loc="upper left")

	# Net Cost
	net_cost_ax = axes[1, 2]
	net_cost_ax.set_title("Net Cost")
	net_cost_ax.set_ylabel("Cost")
	net_cost_ax.set_xlabel("Hour")
	if net_cost_df.empty:
		net_cost_ax.text(0.5, 0.5, "No net cost data", transform=net_cost_ax.transAxes, ha="center", va="center")
	else:
		net_cost_ax.plot(net_cost_df["hour"], net_cost_df["value"], label="Net Cost", linewidth=2, color="tab:purple")
		net_cost_ax.axhline(y=0.0, color="black", linewidth=1, alpha=0.5)
		net_cost_ax.legend(loc="upper left")

	for ax in axes.flatten():
		ax.grid(alpha=0.25)

	fig.tight_layout()
	return fig


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

	st.divider()
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
		selected_single_policy = st.selectbox(
			"Policy",
			options=policies,
			index=0,
			key="single_view_policy",
		)

	st.subheader("Single Household")
	figure = plot_single_household_view(
		player_id=selected_player_id,
		scenario_name=selected_single_scenario,
		policy_name=selected_single_policy,
	)
	st.pyplot(figure, use_container_width=True)
	plt.close(figure)


if __name__ == "__main__":
	main()

