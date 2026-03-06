import streamlit as st
import pandas as pd

from src.sqlite_connection import (
    load_avg_profile as db_load_avg_profile,
)

@st.cache_data(show_spinner=False)
def load_avg_profile(policy_name: str, scenario_name: str, metric: str) -> pd.DataFrame:
    return db_load_avg_profile(
        policy_name=policy_name,
        scenario_name=scenario_name,
        table_name=metric,
    )


def render_aggregate_profile_section(policies: list[str], scenarios: list[str]) -> None:
    st.header("Aggregate Profiles")

    agg_c1, agg_c2, agg_c3 = st.columns([1, 1, 2], gap="large")

    with agg_c1:
        selected_scenario = st.selectbox("Scenario", options=scenarios, index=0)
    with agg_c2:
        selected_metric = st.selectbox("Metric", options=["avg cost", "avg load"], index=0)
    with agg_c3:
        selected_policies = st.multiselect(
            "Policy",
            options=policies,
            default=policies,
        )
        
    series_frames = []

    if not (selected_policies and selected_scenario and selected_metric):
        st.info("Please select a scenario, metric, and at least one policy to view the aggregate profiles.")
        return
    
    for policy_name in selected_policies:
        avg_profile_df = load_avg_profile(policy_name, selected_scenario, selected_metric)
        
        if avg_profile_df.empty:
            st.warning(f"No data found for policy '{policy_name}' in scenario '{selected_scenario}' for metric '{selected_metric}'.")
            continue

        series_frames.append(avg_profile_df)

    st.subheader(f"Aggregate Profile for Policy: {policy_name}")

    if not series_frames:
        st.info("No data available for the selected policies and scenario.")
    else:
        chart_df = pd.concat(series_frames, ignore_index=True)
        pivot_df = chart_df.pivot(index="period", columns="policy", values="avg_value")
        st.line_chart(pivot_df, use_container_width=True)