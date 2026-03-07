import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import streamlit as st

# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.dashboard_2.single_performance.debug_table import render_debug_table
from src.dashboard_2.single_performance.kpi_table import render_kpi_strip
from src.dashboard_2.single_performance.subplots.plot_bess import plot_bess
from src.dashboard_2.single_performance.subplots.plot_ev import plot_ev
from src.dashboard_2.single_performance.subplots.plot_pv import plot_pv
from src.dashboard_2.single_performance.subplots.plot_net_cost import plot_net_cost
from src.dashboard_2.single_performance.subplots.plot_net_load import plot_net_load


from src.sqlite_connection import load_household_result


def _to_optional_bool(value):
	if value is None:
		return None
	return bool(int(value))


def render_single_performance(
        policies: list[str],
        scenarios: list[str],
        household_ids: list[int],
    )-> None:
	st.header("Single View")

	single_c1, single_c2, single_c3 = st.columns([1, 1, 2], gap="large")

	with single_c1:
		selected_household_id = st.selectbox(
			"Household ID",
			options=household_ids,
			index=0,
			key="single_view_household",
		)
	with single_c2:
		selected_scenario_name = st.selectbox(
			"Scenario",
			options=scenarios,
			index=0,
			key="single_view_scenario",
		)
	with single_c3:
		selected_policy_names = st.multiselect(
			"Policy",
			options=policies,
			default=policies[:2] if len(policies) >= 2 else policies,
			key="single_view_policy_multi",
		)

	if not selected_policy_names:
		st.info("Select at least one policy.")
		return

	player_id = selected_household_id
	scenario_name = selected_scenario_name
	policy_names = selected_policy_names

	fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)

	policy_cmap = plt.get_cmap("tab10")
	policy_colors = {
		policy_name: policy_cmap(i % 10)
		for i, policy_name in enumerate(policy_names)
	}

	missed_by_device = {
		"BESS": False,
		"EV1": False,
		"EV2": False,
	}

	for policy_name in policy_names:
		result_df = load_household_result(player_id, scenario_name, policy_name)
		if result_df.empty:
			continue

		result_row = result_df.iloc[0]
		bess_target_met = _to_optional_bool(result_row.get("target_met_bess"))
		ev1_target_met = _to_optional_bool(result_row.get("target_met_ev1"))
		ev2_target_met = _to_optional_bool(result_row.get("target_met_ev2"))

		if bess_target_met is False:
			missed_by_device["BESS"] = True
		if ev1_target_met is False:
			missed_by_device["EV1"] = True
		if ev2_target_met is False:
			missed_by_device["EV2"] = True

	# BESS
	plot_bess(
		ax=axes[0, 0],
		scenario_name=scenario_name,
		player_id=player_id,
		policy_colors=policy_colors,
		missed_deadline=missed_by_device["BESS"],
	)
	
	# EV1
	plot_ev(
		ax=axes[0, 1],
		ev_number="1",
		scenario_name=scenario_name,
		player_id=player_id,
		policy_colors=policy_colors,
		missed_deadline=missed_by_device["EV1"],
	)

	# EV2
	plot_ev(
		ax=axes[0, 2],
		ev_number="2",
		scenario_name=scenario_name,
		player_id=player_id,
		policy_colors=policy_colors,
		missed_deadline=missed_by_device["EV2"],
	)

	# PV
	plot_pv(ax=axes[1, 0], player_id=player_id)

	# Net Load
	plot_net_load(ax=axes[1, 1], scenario_name=scenario_name, player_id=player_id, policy_colors=policy_colors)

	# Net Cost
	plot_net_cost(ax=axes[1, 2], scenario_name=scenario_name, player_id=player_id, policy_colors=policy_colors)

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
	st.pyplot(fig, use_container_width=True)

	st.divider()

	render_kpi_strip(player_id, scenario_name, policy_names)

	st.divider()

	render_debug_table(player_id, scenario_name, policy_names)
