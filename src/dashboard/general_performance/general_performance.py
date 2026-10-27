import streamlit as st
import pandas as pd

from src.simulation.scenarios.scenario import get_scenario_label
from src.sqlite_connection import (
    load_avg_profile as db_load_avg_profile,
    load_run_ids as db_load_run_ids,
    load_scenarios_for_run as db_load_scenarios_for_run,
)


METRICS = {
    "avg_cost": "net_cost",
    "avg_load": "net_load",
    "avg_total_cost": "total_cost",
    "avg_total_consumption": "total_consumption",
}


@st.cache_data(show_spinner=False)
def load_run_ids() -> list[str]:
    return db_load_run_ids()


@st.cache_data(show_spinner=False)
def load_scenarios_for_run(run_id: str) -> list[str]:
    return db_load_scenarios_for_run(run_id)


@st.cache_data(show_spinner=False)
def load_avg_profile(
    policy_name: str,
    scenario_name: str,
    metric: str,
    run_id: str,
) -> pd.DataFrame:
    return db_load_avg_profile(
        policy_name=policy_name,
        scenario_name=scenario_name,
        table_name=metric,
        run_id=run_id,
    )


def render_general_performance(controllers: list[str]) -> None:
    st.header("General Performance")

    run_ids = load_run_ids()
    if not run_ids:
        st.info("No simulation runs are available yet.")
        return

    general_c1, general_c2, general_c3, general_c4 = st.columns(
        [1, 1, 1, 2],
        gap="large",
    )

    with general_c1:
        selected_run_id = st.selectbox("Run ID", options=run_ids)

    run_scenarios = load_scenarios_for_run(selected_run_id)
    if not run_scenarios:
        st.info(f"No scenarios are available for run {selected_run_id}.")
        return

    with general_c2:
        selected_scenario = st.selectbox(
            "Scenario",
            options=run_scenarios,
            index=min(2, len(run_scenarios) - 1),
            format_func=get_scenario_label,
        )
    with general_c3:
        selected_metric = st.selectbox("Metric", options=METRICS, index=0)
    with general_c4:
        selected_policies = st.multiselect(
            "Policy",
            options=controllers,
            default=controllers,
        )
        
    series_frames = []

    if not (selected_policies and selected_scenario and selected_metric):
        st.info("Please select a scenario, metric, and at least one policy to view the aggregate profiles.")
        return
    
    for policy_name in selected_policies:
        avg_profile_df = load_avg_profile(
            policy_name,
            selected_scenario,
            METRICS[selected_metric],
            selected_run_id,
        )
        
        if avg_profile_df.empty:
            st.warning(
                f"No data found for policy '{policy_name}' in scenario "
                f"'{selected_scenario}', metric '{selected_metric}', and run "
                f"'{selected_run_id}'. Older runs may not have per-run time series."
            )
            continue

        series_frames.append(avg_profile_df)

    st.subheader(
        f"Aggregate Profile for Metric: {selected_metric.replace('_', ' ').title()} "
        f"(Run {selected_run_id})"
    )

    if not series_frames:
        st.info("No data available for the selected policies and scenario.")
    else:
        chart_df = pd.concat(series_frames, ignore_index=True)
        pivot_df = chart_df.pivot(
            index="period",
            columns="policy",
            values="value")
        st.line_chart(pivot_df, width="stretch")
