from __future__ import annotations

from src.simulation.controllers.mpc.predictors.shared.make_band import make_band
from src.runtime_conig import RuntimeConfig
from src.simulation.household import Household


# Predictors for base_load, pv_gen

def _forecast_history_average(
    values: list[float],
    horizon: int,
    default: float = 0.0, # default value if no history is available, most profiles start low
) -> list[float]:
    hist_avg = float(sum(values) / len(values)) if values else float(default)
    forecast = [hist_avg] * horizon
    forecast[0] = float(values[-1]) if values else float(default)
    return forecast


def _history_values(household: Household, key: str) -> list[float]:
    history = household.history.get(key, {})
    if not history:
        return []
    return [float(history[timestep]) for timestep in sorted(history)]


def _apply_pv_window_mask(household: Household, series: list[float]) -> list[float]:
    window = RuntimeConfig.PV_GENERATION_WINDOW_OBSERVED
    start_period = int(window["earliest_start"])
    end_period = int(window["latest_end"])
    current_period = int(household.current_timestep)

    masked: list[float] = []
    for idx, value in enumerate(series):
        period = current_period + idx
        if start_period <= period <= end_period:
            masked.append(float(value))
        else:
            masked.append(0.0)
    return masked


def predict_base_load(
    household: Household,
    horizon: int,
    interval_width_fraction: float = 0.1,
) -> dict[str, list[float]]:
    base_load_series = _forecast_history_average(
        values=_history_values(household, "base_load"),
        horizon=horizon,
        default=float(household.base_load),
    )
    base_load_lb, base_load_ub = make_band(base_load_series, interval_width_fraction)

    return {
        "base_load": base_load_series,
        "base_load_lb": base_load_lb,
        "base_load_ub": base_load_ub,
    }


def predict_pv_gen(
    household: Household,
    horizon: int,
    interval_width_fraction: float = 0.1,
) -> dict[str, list[float]]:
    values_since_sunlight = []
    for t, val in sorted(household.history.get("pv_gen", {}).items()):
        if t >= RuntimeConfig.PV_GENERATION_WINDOW_OBSERVED["earliest_start"]:
            values_since_sunlight.append(float(val))

    pv_series = _forecast_history_average(
        values=values_since_sunlight,
        horizon=horizon,
        default=float(household.pv_gen),
    )
    pv_series = _apply_pv_window_mask(household, pv_series)
    pv_lb, pv_ub = make_band(pv_series, interval_width_fraction)

    return {
        "pv_gen": pv_series,
        "pv_gen_lb": pv_lb,
        "pv_gen_ub": pv_ub,
    }
