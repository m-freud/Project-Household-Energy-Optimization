import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import streamlit as st

#~

from src.simulation.scenarios.scenario import get_scenario_label
from src.dashboard.single_performance.debug_table import render_debug_table
from src.dashboard.single_performance.kpi_table import render_kpi_strip
from src.dashboard.single_performance.subplots.plot_bess import plot_bess
from src.dashboard.single_performance.subplots.plot_ev import plot_ev
from src.dashboard.single_performance.subplots.plot_pv import plot_pv
from src.dashboard.single_performance.subplots.plot_base_load import plot_base_load
from src.dashboard.single_performance.subplots.plot_net_load import plot_net_load


from src.sqlite_connection import (
	load_household_result,
	load_household_ids_for_run,
	load_policies_for_run,
	load_run_ids,
	load_scenarios_for_run,
)


def _to_optional_bool(value):
	if value is None:
		return None
	if value != value:
		return None
	return bool(int(value))


def _get_target_met_value(result_row, device_name: str):
	all_key = f"target_met_all_{device_name}"
	final_key = f"target_met_{device_name}"
	all_value = result_row.get(all_key)
	if all_value is not None:
		return _to_optional_bool(all_value)
	return _to_optional_bool(result_row.get(final_key))


def render_single_performance() -> None:
	st.header("Single View")

	run_ids = load_run_ids()
	if not run_ids:
		st.info("No simulation runs are available yet.")
		return

	single_c1, single_c2, single_c3, single_c4 = st.columns(
		[1, 1, 1, 2],
		gap="large",
	)

	with single_c1:
		selected_run_id = st.selectbox(
			"Run ID",
			options=run_ids,
			key="single_view_run_id",
		)

	run_scenarios = load_scenarios_for_run(selected_run_id)
	if not run_scenarios:
		st.info(f"No scenarios are available for run {selected_run_id}.")
		return
	if st.session_state.get("single_view_scenario") not in run_scenarios:
		st.session_state["single_view_scenario"] = run_scenarios[0]

	with single_c2:
		selected_scenario_name = st.selectbox(
			"Scenario",
			options=run_scenarios,
			index=0,
			format_func=get_scenario_label,
			key="single_view_scenario",
		)

	run_household_ids = load_household_ids_for_run(
		selected_run_id,
		selected_scenario_name,
	)
	if not run_household_ids:
		st.info(
			f"No households are available for run {selected_run_id} "
			f"in scenario '{selected_scenario_name}'."
		)
		return
	if st.session_state.get("single_view_household") not in run_household_ids:
		st.session_state["single_view_household"] = run_household_ids[0]

	with single_c3:
		selected_household_id = st.selectbox(
			"Household ID",
			options=run_household_ids,
			index=0,
			key="single_view_household",
		)

	run_policies = load_policies_for_run(selected_run_id, selected_scenario_name)
	if not run_policies:
		st.info(
			f"No policies are available for run {selected_run_id} "
			f"in scenario '{selected_scenario_name}'."
		)
		return
	if "single_view_policy_multi" in st.session_state:
		st.session_state["single_view_policy_multi"] = [
			policy
			for policy in st.session_state["single_view_policy_multi"]
			if policy in run_policies
		]

	with single_c4:
		selected_policy_names = st.multiselect(
			"Policy",
			options=run_policies,
			default=run_policies[:2] if len(run_policies) >= 2 else run_policies,
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
		result_df = load_household_result(
			player_id,
			scenario_name,
			policy_name,
			run_id=selected_run_id,
		)
		if result_df.empty:
			continue

		result_row = result_df.iloc[0]
		bess_target_met = _get_target_met_value(result_row, "bess")
		ev1_target_met = _get_target_met_value(result_row, "ev1")
		ev2_target_met = _get_target_met_value(result_row, "ev2")

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
		run_id=selected_run_id,
	)
	
	# EV1
	plot_ev(
		ax=axes[0, 1],
		ev_number="1",
		scenario_name=scenario_name,
		player_id=player_id,
		policy_colors=policy_colors,
		missed_deadline=missed_by_device["EV1"],
		run_id=selected_run_id,
	)

	# EV2
	plot_ev(
		ax=axes[0, 2],
		ev_number="2",
		scenario_name=scenario_name,
		player_id=player_id,
		policy_colors=policy_colors,
		missed_deadline=missed_by_device["EV2"],
		run_id=selected_run_id,
	)

	# PV
	plot_pv(ax=axes[1, 0], player_id=player_id)

	# Base Load
	plot_base_load(ax=axes[1, 1], scenario_name=scenario_name, player_id=player_id, policy_colors=policy_colors)

	# Net Load
	plot_net_load(
		ax=axes[1, 2],
		scenario_name=scenario_name,
		player_id=player_id,
		policy_colors=policy_colors,
		run_id=selected_run_id,
	)

	policy_handles = [
		Line2D([0], [0], color=policy_colors[policy_name], linewidth=2, label=policy_name)
		for policy_name in policy_names
	]

	style_handles = [
		Line2D([0], [0], color="tab:red", linestyle="--", linewidth=1.5, label="Target SOC"),
		Line2D([0], [0], color="tab:red", marker="x", linestyle="None", markersize=7, label="SOC Target Checkpoint"),
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

	render_kpi_strip(player_id, scenario_name, policy_names, selected_run_id)

	st.divider()

	render_debug_table(player_id, scenario_name, policy_names, selected_run_id)
