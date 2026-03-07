import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
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


from src.sqlite_connection import load_household_result, load_series
from src.simulation.scenarios.scenario import get_scenario_value


def render_single_performance(
        player_id: int,
        scenario_name: str,
        policy_names: list[str],
    )-> None:

	fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)

	policy_cmap = plt.get_cmap("tab10")
	policy_colors = {
		policy_name: policy_cmap(i % 10)
		for i, policy_name in enumerate(policy_names)
	}

	# BESS
	plot_bess(ax=axes[0, 0], scenario_name=scenario_name, player_id=player_id, policy_colors=policy_colors)
	
	# EV1
	plot_ev(ax=axes[0, 1], ev_number="1", scenario_name=scenario_name, player_id=player_id, policy_colors=policy_colors)

	# EV2
	plot_ev(ax=axes[0, 2], ev_number="2", scenario_name=scenario_name, player_id=player_id, policy_colors=policy_colors)

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

	st.balloons()
