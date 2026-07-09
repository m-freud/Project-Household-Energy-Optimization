from __future__ import annotations

import math
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.simulation.controllers.mpc.predictors.moving_average_predictor import MovingAveragePredictor
from src.simulation.controllers.mpc.predictors.moving_average_predictor2 import MovingAveragePredictor2
from src.simulation.controllers.mpc.predictors.moving_average_predictor3 import MovingAveragePredictor3
from src.simulation.controllers.mpc.predictors.oracle_predictor import OraclePredictor
from src.simulation.scenarios.scenario import SCENARIOS_BY_NAME
from src.simulation.simulation import Simulation
from src.sqlite_connection import create_sqlite_connection


PREDICTOR_OPTIONS = {
    "oracle": "oracle",
    "ma1": "ma1",
    "ma2": "ma2",
    "ma3": "ma3",
}

PROFILE_OPTIONS = [
    "base_load",
    "pv_gen",
    "ev1_load",
    "ev2_load",
    "ev1_at_home",
    "ev1_at_charging_station",
    "ev2_at_home",
    "ev2_at_charging_station",
    "buy_price",
    "sell_price",
    "ev1_buy_price",
    "ev2_buy_price",
    "ev1_max_charge",
    "ev2_max_charge",
]


def _build_predictor(
    predictor_name: str,
    ma1_window: int,
    ma2_short: int,
    ma2_long: int,
    ma2_weight: float,
    ma3_short: int,
    ma3_long: int,
    ma3_weight: float,
    ma3_interval_width: float,
) -> BasePredictor:
    if predictor_name == "ma1":
        return MovingAveragePredictor(window_size=ma1_window)
    if predictor_name == "ma2":
        return MovingAveragePredictor2(
            short_window_size=ma2_short,
            long_window_size=ma2_long,
            short_weight=ma2_weight,
        )
    if predictor_name == "ma3":
        return MovingAveragePredictor3(
            short_window_size=ma3_short,
            long_window_size=ma3_long,
            short_weight=ma3_weight,
            interval_width_fraction=ma3_interval_width,
        )
    return OraclePredictor()


@st.cache_data(show_spinner=False)
def _compute_snapshot(
    player_id: int,
    scenario_name: str,
    predictor_name: str,
    timestep: int,
    horizon: int,
    ma1_window: int,
    ma2_short: int,
    ma2_long: int,
    ma2_weight: float,
    ma3_short: int,
    ma3_long: int,
    ma3_weight: float,
    ma3_interval_width: float,
) -> tuple[dict[str, list[float]], dict[str, list[float]], int]:
    scenario = SCENARIOS_BY_NAME[scenario_name]

    predictor = _build_predictor(
        predictor_name=predictor_name,
        ma1_window=ma1_window,
        ma2_short=ma2_short,
        ma2_long=ma2_long,
        ma2_weight=ma2_weight,
        ma3_short=ma3_short,
        ma3_long=ma3_long,
        ma3_weight=ma3_weight,
        ma3_interval_width=ma3_interval_width,
    )

    connection = create_sqlite_connection()
    try:
        sim = Simulation(connection, ensure_schema=False)
        # create_household only needs scenario + start_time from this object.
        run_context = SimpleNamespace(scenario=scenario, start_time=1)
        household = sim.create_household(player_id, run_context)

        current_timestep = max(1, min(96, int(timestep)))
        for period in range(1, current_timestep + 1):
            sim.current_timestep = period
            sim.update_household_inputs(household)
            household.update_history()

        effective_horizon = max(1, min(int(horizon), 96 - current_timestep + 1))
        predicted = predictor.predict(household, scenario, effective_horizon)
        actual = household.oracle_profiles
        return actual, predicted, effective_horizon
    finally:
        connection.close()


def _full_day_prediction_series(
    predicted_values: list[float],
    timestep: int,
    horizon: int,
) -> list[float]:
    full = [math.nan] * 96
    start_idx = max(0, min(95, int(timestep) - 1))
    stop_idx = min(96, start_idx + int(horizon))
    for idx in range(start_idx, stop_idx):
        pred_idx = idx - start_idx
        if pred_idx < len(predicted_values):
            full[idx] = float(predicted_values[pred_idx])
    return full


def render_prediction_explorer(
    household_ids: list[int],
    scenarios: list[str],
) -> None:
    st.header("Prediction Explorer")

    c1, c2, c3, c4 = st.columns([1, 1, 2, 3], gap="large")

    with c1:
        selected_household_id = st.selectbox(
            "Household",
            options=household_ids,
            index=0,
            key="pred_household",
        )
    with c2:
        selected_scenario = st.selectbox(
            "Scenario",
            options=scenarios,
            index=0,
            key="pred_scenario",
        )
    with c3:
        selected_predictors = st.multiselect(
            "Predictors",
            options=list(PREDICTOR_OPTIONS.keys()),
            default=["ma3"],
            key="pred_predictors",
            format_func=lambda value: PREDICTOR_OPTIONS[value],
        )
    with c4:
        selected_profiles = st.multiselect(
            "Profiles",
            options=PROFILE_OPTIONS,
            default=["base_load"],
            key="pred_profiles",
        )

    if not selected_predictors:
        st.info("Select at least one predictor.")
        return
    if not selected_profiles:
        st.info("Select at least one profile.")
        return

    with st.expander("Predictor hyperparameters", expanded=False):
        p1, p2, p3 = st.columns(3)
        with p1:
            ma1_window = int(st.number_input("ma1 window", min_value=1, max_value=96, value=12, step=1))
        with p2:
            ma2_short = int(st.number_input("ma2 short", min_value=1, max_value=96, value=7, step=1))
            ma2_weight = float(st.slider("ma2 short weight", min_value=0.0, max_value=1.0, value=0.7, step=0.05))
        with p3:
            ma2_long = int(st.number_input("ma2 long", min_value=1, max_value=96, value=48, step=1))

        q1, q2, q3 = st.columns(3)
        with q1:
            ma3_short = int(st.number_input("ma3 short", min_value=1, max_value=96, value=7, step=1))
        with q2:
            ma3_long = int(st.number_input("ma3 long", min_value=1, max_value=96, value=48, step=1))
            ma3_weight = float(st.slider("ma3 short weight", min_value=0.0, max_value=1.0, value=0.7, step=0.05))
        with q3:
            ma3_interval_width = float(st.slider("ma3 interval width", min_value=0.0, max_value=0.5, value=0.1, step=0.01))

    t_key = "pred_timestep"
    if t_key not in st.session_state:
        st.session_state[t_key] = 1

    nav1, nav2, nav3, nav4 = st.columns([1, 1, 5, 2])
    with nav1:
        if st.button("<- Prev", width="stretch"):
            st.session_state[t_key] = max(1, int(st.session_state[t_key]) - 1)
    with nav2:
        if st.button("Next ->", width="stretch"):
            st.session_state[t_key] = min(96, int(st.session_state[t_key]) + 1)
    with nav3:
        st.session_state[t_key] = int(
            st.slider(
                "Timestep",
                min_value=1,
                max_value=96,
                value=int(st.session_state[t_key]),
            )
        )
    with nav4:
        horizon = int(st.number_input("Horizon", min_value=1, max_value=96, value=24, step=1))

    timestep = int(st.session_state[t_key])

    snapshots: dict[str, tuple[dict[str, list[float]], dict[str, list[float]], int]] = {}
    for predictor_name in selected_predictors:
        snapshots[predictor_name] = _compute_snapshot(
            player_id=int(selected_household_id),
            scenario_name=str(selected_scenario),
            predictor_name=str(predictor_name),
            timestep=timestep,
            horizon=horizon,
            ma1_window=ma1_window,
            ma2_short=ma2_short,
            ma2_long=ma2_long,
            ma2_weight=ma2_weight,
            ma3_short=ma3_short,
            ma3_long=ma3_long,
            ma3_weight=ma3_weight,
            ma3_interval_width=ma3_interval_width,
        )

    actual_ref = snapshots[selected_predictors[0]][0]
    min_effective_horizon = min(value[2] for value in snapshots.values())

    valid_profiles = [profile for profile in selected_profiles if actual_ref.get(profile, [])]
    if not valid_profiles:
        st.warning("None of the selected profiles had data for this household.")
        return

    missing_profiles = [profile for profile in selected_profiles if profile not in valid_profiles]
    if missing_profiles:
        st.info(f"Skipped profiles with no data: {', '.join(missing_profiles)}")

    fig, axes = plt.subplots(
        nrows=len(valid_profiles),
        ncols=1,
        figsize=(14, max(4.5, 3.2 * len(valid_profiles))),
        sharex=True,
    )
    if len(valid_profiles) == 1:
        axes = [axes]

    predictor_cmap = plt.get_cmap("tab10")
    predictor_colors = {
        predictor_name: predictor_cmap(index % 10)
        for index, predictor_name in enumerate(selected_predictors)
    }

    for axis, profile_name in zip(axes, valid_profiles):
        actual_values = [float(value) for value in actual_ref.get(profile_name, [])]
        x = np.arange(1, len(actual_values) + 1)
        axis.plot(x, actual_values, color="black", linewidth=1.8, label="actual")

        for predictor_name in selected_predictors:
            predicted = snapshots[predictor_name][1]
            predicted_values = [float(value) for value in predicted.get(profile_name, [])]
            full_day_pred = _full_day_prediction_series(predicted_values, timestep, min_effective_horizon)

            axis.plot(
                x,
                full_day_pred,
                color=predictor_colors[predictor_name],
                linewidth=1.9,
                linestyle="--",
                label=f"pred ({predictor_name})",
            )

            if predictor_name == "ma3" and profile_name in {"base_load", "pv_gen"}:
                lb_key = f"{profile_name}_lb"
                ub_key = f"{profile_name}_ub"
                lb = predicted.get(lb_key, [])
                ub = predicted.get(ub_key, [])
                if lb and ub:
                    full_day_lb = _full_day_prediction_series([float(v) for v in lb], timestep, min_effective_horizon)
                    full_day_ub = _full_day_prediction_series([float(v) for v in ub], timestep, min_effective_horizon)
                    axis.fill_between(x, full_day_lb, full_day_ub, color=predictor_colors[predictor_name], alpha=0.12)

        axis.axvline(float(timestep), color="red", linewidth=1.8, linestyle="-", label="current timestep")
        axis.set_title(profile_name)
        axis.set_ylabel("Value")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", ncols=2)

    axes[-1].set_xlabel("Timestep (1-96)")
    fig.suptitle(
        f"Prediction Explorer | player {selected_household_id} | scenario {selected_scenario}",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    st.pyplot(fig, width="stretch")

    info1, info2, info3 = st.columns(3)
    with info1:
        st.metric("Current timestep", timestep)
    with info2:
        st.metric("Effective horizon", min_effective_horizon)
    with info3:
        metric_profile = valid_profiles[0]
        metric_predictor = selected_predictors[0]
        metric_predicted = [float(value) for value in snapshots[metric_predictor][1].get(metric_profile, [])]
        metric_actual = [float(value) for value in actual_ref.get(metric_profile, [])]
        if metric_predicted:
            first_pred = metric_predicted[0]
            first_actual = metric_actual[timestep - 1] if 0 <= timestep - 1 < len(metric_actual) else math.nan
            first_error = float(first_pred) - float(first_actual)
            st.metric("First-step error", f"{first_error:.4f}")
        else:
            st.metric("First-step error", "n/a")

    with st.expander("Prediction preview table", expanded=False):
        rows = []
        for profile_name in valid_profiles:
            actual_values = [float(value) for value in actual_ref.get(profile_name, [])]
            for predictor_name in selected_predictors:
                predicted_values = [float(value) for value in snapshots[predictor_name][1].get(profile_name, [])]
                preview_h = min(8, len(predicted_values))
                for i in range(preview_h):
                    period = timestep + i
                    actual_val = actual_values[period - 1] if period - 1 < len(actual_values) else math.nan
                    pred_val = predicted_values[i]
                    rows.append(
                        {
                            "profile": profile_name,
                            "predictor": predictor_name,
                            "period": period,
                            "actual": float(actual_val),
                            "predicted": float(pred_val),
                            "error": float(pred_val) - float(actual_val),
                        }
                    )

        if rows:
            st.dataframe(rows, width="stretch")
        else:
            st.info("No prediction values available for the selected combinations.")
