from __future__ import annotations

from datetime import datetime
from functools import partial
import math
import re
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.config import Config
from src.simulation.controllers.mpc.predictors.hybrid_ma_predictor import HybridMAPredictor
from src.simulation.controllers.mpc.mpc_controller import MPCController
from src.simulation.controllers.mpc.predictors.oracle_predictor import OraclePredictor
from src.simulation.run_context import RunContext
from src.simulation.scenarios.scenario import scenarios as scenario_catalog
from src.simulation.scenarios.scenario import default_scenario
from src.simulation.simulation import Simulation
from src.sqlite_connection import create_sqlite_connection, load_source_avg_profile


PREDICTOR_OPTIONS = {
    "oracle": "oracle",
    "hybrid_ma": "hybrid_ma_controller",
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
    ma3_short: int,
    ma3_long: int,
    ma3_weight: float,
    ma3_interval_width: float,
    ma3_persistence_mode: str,
    ma3_persistence_range: int,
    ma3_persistence_constant_alpha: float,
    source_average_beta: float,
    ma3_trend_weight: float,
    ma3_trend_window: int,
    ma3_trend_range: int,
):
    if predictor_name == "hybrid_ma":
        return HybridMAPredictor(
            short_window_size=ma3_short,
            long_window_size=ma3_long,
            short_weight=ma3_weight,
            conf_interval_frct=ma3_interval_width,
            persistence_mode=ma3_persistence_mode,
            persistence_range=ma3_persistence_range,
            persistence_constant_alpha=ma3_persistence_constant_alpha,
            source_average_beta=source_average_beta,
            trend_weight=ma3_trend_weight,
            trend_window=ma3_trend_window,
            trend_range=ma3_trend_range,
        )
    return OraclePredictor()


@st.cache_data(show_spinner=False)
def _compute_snapshot(
    player_id: int,
    predictor_name: str,
    timestep: int,
    horizon: int,
    ma3_short: int,
    ma3_long: int,
    ma3_weight: float,
    ma3_interval_width: float,
    ma3_persistence_mode: str,
    ma3_persistence_range: int,
    ma3_persistence_constant_alpha: float,
    source_average_beta: float,
    ma3_trend_weight: float,
    ma3_trend_window: int,
    ma3_trend_range: int,
) -> tuple[dict[str, list[float]], dict[str, list[float]], int]:
    scenario = default_scenario

    predictor = _build_predictor(
        predictor_name=predictor_name,
        ma3_short=ma3_short,
        ma3_long=ma3_long,
        ma3_weight=ma3_weight,
        ma3_interval_width=ma3_interval_width,
        ma3_persistence_mode=ma3_persistence_mode,
        ma3_persistence_range=ma3_persistence_range,
        ma3_persistence_constant_alpha=ma3_persistence_constant_alpha,
        source_average_beta=source_average_beta,
        ma3_trend_weight=ma3_trend_weight,
        ma3_trend_window=ma3_trend_window,
        ma3_trend_range=ma3_trend_range,
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


@st.cache_data(show_spinner=False)
def _load_source_avg_curve(table_name: str) -> list[float]:
    df = load_source_avg_profile(table_name)
    if df.empty:
        return []
    return [float(value) for value in df["value"].tolist()]

def _slugify_name(value: str) -> str:
    compact = re.sub(r"\s+", "_", value.strip().lower())
    compact = re.sub(r"[^a-z0-9_\-]", "", compact)
    return compact.strip("_-")


def _format_float_token(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}".replace(".", "p")


def _build_default_policy_name(
    predictor_name: str,
    ma3_short: int,
    ma3_long: int,
    ma3_weight: float,
    ma3_interval_width: float,
    ma3_persistence_mode: str,
    ma3_persistence_range: int,
    ma3_persistence_constant_alpha: float,
    source_average_beta: float,
    ma3_trend_weight: float,
    ma3_trend_window: int,
    ma3_trend_range: int,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if predictor_name == "hybrid_ma":
        return (
            f"mpc_hybrid_ma_sw{int(ma3_short)}_lw{int(ma3_long)}"
            f"_w{_format_float_token(ma3_weight)}"
            f"_ci{_format_float_token(ma3_interval_width)}"
            f"_pm{_slugify_name(ma3_persistence_mode) or 'exp'}"
            f"_pr{int(ma3_persistence_range)}"
            f"_pa{_format_float_token(ma3_persistence_constant_alpha)}"
            f"_tw{_format_float_token(ma3_trend_weight)}"
            f"_twin{int(ma3_trend_window)}"
            f"_tr{int(ma3_trend_range)}"
            f"_sa{_format_float_token(source_average_beta)}"
            f"_{timestamp}"
        )
    return f"mpc_oracle_{timestamp}"


def _build_mpc_controller_factory(
    policy_name: str,
    horizon: int,
    predictor_name: str,
    ma3_short: int,
    ma3_long: int,
    ma3_weight: float,
    ma3_interval_width: float,
    ma3_persistence_mode: str,
    ma3_persistence_range: int,
    ma3_persistence_constant_alpha: float,
    source_average_beta: float,
    ma3_trend_weight: float,
    ma3_trend_window: int,
    ma3_trend_range: int,
):
    return partial(
        _build_mpc_controller,
        policy_name=policy_name,
        horizon=horizon,
        predictor_name=predictor_name,
        ma3_short=ma3_short,
        ma3_long=ma3_long,
        ma3_weight=ma3_weight,
        ma3_interval_width=ma3_interval_width,
        ma3_persistence_mode=ma3_persistence_mode,
        ma3_persistence_range=ma3_persistence_range,
        ma3_persistence_constant_alpha=ma3_persistence_constant_alpha,
        source_average_beta=source_average_beta,
        ma3_trend_weight=ma3_trend_weight,
        ma3_trend_window=ma3_trend_window,
        ma3_trend_range=ma3_trend_range,
    )


def _build_mpc_controller(
    household,
    scenario,
    *,
    policy_name: str,
    horizon: int,
    predictor_name: str,
    ma3_short: int,
    ma3_long: int,
    ma3_weight: float,
    ma3_interval_width: float,
    ma3_persistence_mode: str,
    ma3_persistence_range: int,
    ma3_persistence_constant_alpha: float,
    source_average_beta: float,
    ma3_trend_weight: float,
    ma3_trend_window: int,
    ma3_trend_range: int,
):
    base_predictor = _build_predictor(
        predictor_name=predictor_name,
        ma3_short=ma3_short,
        ma3_long=ma3_long,
        ma3_weight=ma3_weight,
        ma3_interval_width=ma3_interval_width,
        ma3_persistence_mode=ma3_persistence_mode,
        ma3_persistence_range=ma3_persistence_range,
        ma3_persistence_constant_alpha=ma3_persistence_constant_alpha,
        source_average_beta=source_average_beta,
        ma3_trend_weight=ma3_trend_weight,
        ma3_trend_window=ma3_trend_window,
        ma3_trend_range=ma3_trend_range,
    )

    return MPCController(
        name=policy_name,
        household=household,
        scenario=scenario,
        horizon=int(horizon),
        predictor=base_predictor,
        duration_hours=float(Config.DURATION_TIMESTEP),
    )


def render_prediction_explorer(
    household_ids: list[int],
) -> None:
    st.header("Prediction Explorer")

    c1, c2, c3 = st.columns([1, 2, 3], gap="large")

    with c1:
        selected_household_id = st.selectbox(
            "Household",
            options=household_ids,
            index=0,
            key="pred_household",
        )
    with c2:
        selected_predictors = st.multiselect(
            "Predictors",
            options=list(PREDICTOR_OPTIONS.keys()),
            default=["hybrid_ma"],
            key="pred_predictors",
            format_func=lambda value: PREDICTOR_OPTIONS[value],
        )
    with c3:
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
        s1, d1, s2, d2, s3 = st.columns([1, 0.05, 1, 0.05, 1], gap="medium")

        with s1:
            st.markdown("**Moving average**")
            short_row = st.columns([5, 1], gap="small")
            with short_row[0]:
                ma3_short = int(st.number_input("short window", min_value=0, max_value=96, value=0, step=1))
            with short_row[1]:
                st.markdown("<div style='padding-top: 0.35rem; margin-bottom: -0.25rem;'>show</div>", unsafe_allow_html=True)
                show_short_window = st.checkbox(
                    "show short window",
                    value=False,
                    key="show_short_window",
                    label_visibility="collapsed",
                )

            long_row = st.columns([5, 1], gap="small")
            with long_row[0]:
                ma3_long = int(st.number_input("long window", min_value=1, max_value=96, value=96, step=1))
            with long_row[1]:
                st.markdown("<div style='padding-top: 0.35rem; margin-bottom: -0.25rem;'>show</div>", unsafe_allow_html=True)
                show_long_window = st.checkbox(
                    "show long window",
                    value=True,
                    key="show_long_window",
                    label_visibility="collapsed",
                )

            ma3_weight = float(st.slider("short weight", min_value=0.0, max_value=1.0, value=0.0, step=0.05))

        with d1:
            st.markdown(
                "<div style='border-left: 1px solid rgba(128, 128, 128, 0.35); height: 18rem; margin: 0 auto;'></div>",
                unsafe_allow_html=True,
            )

        with s2:
            st.markdown("**Value persistence**")
            ma3_persistence_mode = st.selectbox(
                "persistence mode",
                options=["exponential", "linear", "constant", "none"],
                index=2,
            )
            persistence_row = st.columns([5, 1], gap="small")
            with persistence_row[0]:
                ma3_persistence_range = int(
                    st.number_input(
                        "persistence range",
                        min_value=0,
                        max_value=96,
                        value=0,
                        step=1,
                        disabled=ma3_persistence_mode == "none",
                    )
                )
            with persistence_row[1]:
                st.markdown("<div style='padding-top: 0.35rem; margin-bottom: -0.25rem;'>show</div>", unsafe_allow_html=True)
                show_persistence_range = st.checkbox(
                    "show persistence range",
                    value=False,
                    key="show_persistence_range",
                    disabled=ma3_persistence_mode == "none",
                    label_visibility="collapsed",
                )

            ma3_persistence_constant_alpha = float(
                st.slider(
                    "constant alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    disabled=ma3_persistence_mode != "constant",
                )
            )

        with d2:
            st.markdown(
                "<div style='border-left: 1px solid rgba(128, 128, 128, 0.35); height: 18rem; margin: 0 auto;'></div>",
                unsafe_allow_html=True,
            )

        with s3:
            st.markdown("**Trend persistence**")
            trend_window_row = st.columns([5, 1], gap="small")
            with trend_window_row[0]:
                ma3_trend_window = int(
                    st.number_input(
                        "trend window",
                        min_value=0,
                        max_value=96,
                        value=0,
                        step=1,
                    )
                )
            with trend_window_row[1]:
                st.markdown("<div style='padding-top: 0.35rem; margin-bottom: -0.25rem;'>show</div>", unsafe_allow_html=True)
                show_trend_window = st.checkbox(
                    "show trend window",
                    value=False,
                    key="show_trend_window",
                    label_visibility="collapsed",
                )

            trend_range_row = st.columns([5, 1], gap="small")
            with trend_range_row[0]:
                ma3_trend_range = int(
                    st.number_input(
                        "trend persistence range",
                        min_value=0,
                        max_value=96,
                        value=0,
                        step=1,
                    )
                )
            with trend_range_row[1]:
                st.markdown("<div style='padding-top: 0.35rem; margin-bottom: -0.25rem;'>show</div>", unsafe_allow_html=True)
                show_trend_range = st.checkbox(
                    "show trend range",
                    value=False,
                    key="show_trend_range",
                    label_visibility="collapsed",
                )

            ma3_trend_weight = float(
                st.slider(
                    "trend weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                )
            )

        st.markdown("---")

        b1, bd1, b2, bd2, b3 = st.columns([1, 0.05, 1, 0.05, 1], gap="medium")

        with b1:
            ma3_interval_width = float(
                st.slider("confidence interval", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
            )

        with bd1:
            st.markdown(
                "<div style='border-left: 1px solid rgba(128, 128, 128, 0.35); height: 6rem; margin: 0 auto;'></div>",
                unsafe_allow_html=True,
            )

        with b2:
            source_average_beta = float(
                st.slider(
                    "source-average beta",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    help="Applied to MA3 prediction only: 0.0 = no source-average blend, 1.0 = source average only",
                )
            )
            show_source_average = st.checkbox("Show all-household source average", value=True)

        with bd2:
            st.markdown(
                "<div style='border-left: 1px solid rgba(128, 128, 128, 0.35); height: 6rem; margin: 0 auto;'></div>",
                unsafe_allow_html=True,
            )

        with b3:
            show_pv_unavailable_shadow = st.checkbox("Show PV unavailable shadow", value=True)

    t_key = "pred_timestep"
    if t_key not in st.session_state:
        st.session_state[t_key] = 42

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
        horizon = int(st.number_input("Horizon", min_value=1, max_value=96, value=96, step=1))

    timestep = int(st.session_state[t_key])

    # Display window/range overlays using the Hybrid MA configuration.
    if "hybrid_ma" in selected_predictors:
        display_short_window = int(ma3_short)
        display_long_window = int(ma3_long)
        display_persistence_range = int(ma3_persistence_range)
        display_trend_window = int(ma3_trend_window)
        display_trend_range = int(ma3_trend_range)
    else:
        display_short_window = 0
        display_long_window = 0
        display_persistence_range = 0
        display_trend_window = 0
        display_trend_range = 0

    snapshots: dict[str, tuple[dict[str, list[float]], dict[str, list[float]], int]] = {}
    for predictor_name in selected_predictors:
        snapshots[predictor_name] = _compute_snapshot(
            player_id=int(selected_household_id),
            predictor_name=str(predictor_name),
            timestep=timestep,
            horizon=horizon,
            ma3_short=ma3_short,
            ma3_long=ma3_long,
            ma3_weight=ma3_weight,
            ma3_interval_width=ma3_interval_width,
            ma3_persistence_mode=ma3_persistence_mode,
            ma3_persistence_range=ma3_persistence_range,
            ma3_persistence_constant_alpha=ma3_persistence_constant_alpha,
            source_average_beta=source_average_beta,
            ma3_trend_weight=ma3_trend_weight,
            ma3_trend_window=ma3_trend_window,
            ma3_trend_range=ma3_trend_range,
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

        if show_pv_unavailable_shadow and profile_name == "pv_gen":
            pv_window = getattr(Config, "PV_GENERATION_WINDOW_OBSERVED", None) or getattr(
                Config,
                "PV_GENERATION_WINDOW_ALLOWED",
                None,
            )
            if pv_window:
                pv_start = max(1, min(96, int(pv_window.get("earliest_start", 1))))
                pv_end = max(1, min(96, int(pv_window.get("latest_end", 96))))

                if pv_start > 1:
                    axis.axvspan(1.0, float(pv_start), color="gray", alpha=0.10, label="pv unavailable")
                if pv_end < 96:
                    axis.axvspan(float(pv_end), 96.0, color="gray", alpha=0.10)

        source_avg_values: list[float] = []
        if show_source_average and profile_name in {"base_load", "pv_gen"}:
            source_avg_values = _load_source_avg_curve(profile_name)
            if source_avg_values:
                source_x = np.arange(1, len(source_avg_values) + 1)
                axis.plot(
                    source_x,
                    source_avg_values,
                    color="tab:purple",
                    linewidth=1.6,
                    linestyle=":",
                    label="source avg (all households)",
                )

        if show_long_window and display_long_window > 0:
            long_start = max(1, timestep - display_long_window + 1)
            axis.axvspan(
                float(long_start),
                float(timestep),
                color="tab:orange",
                alpha=0.08,
                label=f"long window ({display_long_window})",
            )

        if show_short_window and display_short_window > 0:
            short_start = max(1, timestep - display_short_window + 1)
            axis.axvspan(
                float(short_start),
                float(timestep),
                color="tab:green",
                alpha=0.15,
                label=f"short window ({display_short_window})",
            )

        if show_trend_window and display_trend_window > 0:
            trend_window_start = max(1, timestep - display_trend_window + 1)
            axis.axvspan(
                float(trend_window_start),
                float(timestep),
                color="tab:blue",
                alpha=0.08,
                label=f"trend window ({display_trend_window})",
            )

        if show_persistence_range and display_persistence_range > 0:
            persistence_end = min(96, timestep + display_persistence_range - 1)
            axis.axvspan(
                float(timestep),
                float(persistence_end),
                color="tab:brown",
                alpha=0.08,
                label=f"persistence range ({display_persistence_range})",
            )

        if show_trend_range and display_trend_range > 0:
            # Trend target uses min(step_index - 1, trend_range), so the change
            # reaches its final step one period later than persistence range.
            trend_range_end = min(96, timestep + display_trend_range)
            axis.axvspan(
                float(timestep),
                float(trend_range_end),
                color="tab:cyan",
                alpha=0.08,
                label=f"trend range ({display_trend_range})",
            )

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

            if predictor_name == "hybrid_ma" and profile_name in {"base_load", "pv_gen"}:
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
        f"Prediction Explorer | player {selected_household_id}",
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

    st.markdown("---")
    st.subheader("Run MPC With This Predictor")

    if len(selected_predictors) != 1:
        st.info("Select exactly one predictor to run MPC and save results.")
        return

    selected_run_predictor = str(selected_predictors[0])
    generated_policy_name = _build_default_policy_name(
        predictor_name=selected_run_predictor,
        ma3_short=ma3_short,
        ma3_long=ma3_long,
        ma3_weight=ma3_weight,
        ma3_interval_width=ma3_interval_width,
        ma3_persistence_mode=ma3_persistence_mode,
        ma3_persistence_range=ma3_persistence_range,
        ma3_persistence_constant_alpha=ma3_persistence_constant_alpha,
        source_average_beta=source_average_beta,
        ma3_trend_weight=ma3_trend_weight,
        ma3_trend_window=ma3_trend_window,
        ma3_trend_range=ma3_trend_range,
    )

    name_help = (
        "Leave empty to auto-generate a policy name that includes predictor settings and a timestamp."
    )
    run_name_input = st.text_input(
        "Policy name",
        value="",
        placeholder=generated_policy_name,
        help=name_help,
        key="pred_run_policy_name",
    )
    effective_policy_name = _slugify_name(run_name_input) or generated_policy_name

    st.caption(f"Will run policy: {effective_policy_name}")
    st.caption(f"Scenarios: {len(scenario_catalog)} | Households: {len(household_ids)}")

    run_button_label = "Run MPC With This Predictor"
    if st.button(run_button_label, type="primary", width="stretch"):
        total_scenarios = len(scenario_catalog)
        household_id_list = [int(player_id) for player_id in household_ids]
        total_households = len(household_id_list)
        total_jobs = total_scenarios * total_households
        if total_scenarios <= 0 or total_households <= 0:
            st.warning("No household/scenario combinations available to run.")
            return

        progress = st.progress(0.0, text="Preparing MPC runs...")
        status = st.empty()
        created_run_ids: list[str] = []
        completed_jobs = 0

        connection = create_sqlite_connection()
        try:
            # Fresh databases may not have results schema yet.
            sim = Simulation(connection, ensure_schema=True)
            controller_factory = _build_mpc_controller_factory(
                policy_name=effective_policy_name,
                horizon=96,
                predictor_name=selected_run_predictor,
                ma3_short=ma3_short,
                ma3_long=ma3_long,
                ma3_weight=ma3_weight,
                ma3_interval_width=ma3_interval_width,
                ma3_persistence_mode=ma3_persistence_mode,
                ma3_persistence_range=ma3_persistence_range,
                ma3_persistence_constant_alpha=ma3_persistence_constant_alpha,
                source_average_beta=source_average_beta,
                ma3_trend_weight=ma3_trend_weight,
                ma3_trend_window=ma3_trend_window,
                ma3_trend_range=ma3_trend_range,
            )

            for scenario in scenario_catalog:
                run_context = RunContext(
                    controller_factory=controller_factory,
                    controller_name=effective_policy_name,
                    scenario=scenario,
                    start_time=1,
                )
                created_run_ids.append(run_context.run_id)

                status.info(f"Running scenario: {scenario.name}")
                chunk_size = max(1, min(24, total_households // 4 if total_households > 4 else total_households))
                for start_idx in range(0, total_households, chunk_size):
                    chunk_households = household_id_list[start_idx : start_idx + chunk_size]
                    sim.run_all_households(
                        run_context,
                        household_ids=chunk_households,
                        parallel_households=True,
                        parallel_workers=6,
                    )
                    completed_jobs += len(chunk_households)
                    progress.progress(
                        float(completed_jobs) / float(total_jobs),
                        text=f"Running {effective_policy_name}: {completed_jobs}/{total_jobs}",
                    )

        except Exception as exc:
            progress.empty()
            status.error(f"MPC run failed: {exc}")
            return
        finally:
            connection.close()

        progress.progress(1.0, text="Finished. Results were saved to SQLite.")
        status.success(
            "Saved run for all scenarios. "
            f"Policy={effective_policy_name}, run_ids={', '.join(sorted(set(created_run_ids)))}"
        )

        # Ensure dashboard-level cached queries see the newly stored run.
        st.cache_data.clear()

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
