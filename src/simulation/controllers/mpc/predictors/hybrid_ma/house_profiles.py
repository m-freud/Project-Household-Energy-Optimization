from __future__ import annotations

from src.config import Config
from src.simulation.household import Household
from .moving_average import forecast_moving_average

# Predictors for base_load, pv_gen

def _history_values(household: Household, key: str) -> list[float]:
    history = household.history.get(key, {})
    if not history:
        return []
    return [float(history[timestep]) for timestep in sorted(history)]


def _make_band(series: list[float], width_fraction: float) -> tuple[list[float], list[float]]:
    width_fraction = max(0.0, float(width_fraction))
    lower = [max(0.0, value * (1.0 - width_fraction)) for value in series]
    upper = [max(0.0, value * (1.0 + width_fraction)) for value in series]
    return lower, upper


def _apply_pv_window_mask(household: Household, series: list[float]) -> list[float]:
    window = Config.PV_GENERATION_WINDOW_OBSERVED
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
    window_size: int = 48,
    persistence_range: int = 1,
    interval_width_fraction: float = 0.0,
) -> dict[str, list[float]]:
    base_series = forecast_moving_average(
        values=_history_values(household, "base_load"),
        horizon=horizon,
        window_size=window_size,
        default=float(household.base_load),
        persistence_range=persistence_range,
    )
    base_lb, base_ub = _make_band(base_series, interval_width_fraction)

    return {
        "base_load": base_series,
        "base_load_lb": base_lb,
        "base_load_ub": base_ub,
    }


def predict_pv_gen(
    household: Household,
    horizon: int,
    window_size: int = 96,
    persistence_range: int = 1,
    interval_width_fraction: float = 0.0,
) -> dict[str, list[float]]:
    pv_series = forecast_moving_average(
        values=_history_values(household, "pv_gen"),
        horizon=horizon,
        window_size=window_size,
        default=float(household.pv_gen),
        persistence_range=persistence_range,
    )
    pv_series = _apply_pv_window_mask(household, pv_series)
    pv_lb, pv_ub = _make_band(pv_series, interval_width_fraction)

    return {
        "pv_gen": pv_series,
        "pv_gen_lb": pv_lb,
        "pv_gen_ub": pv_ub,
    }


def predict_house_profiles(
    household: Household,
    horizon: int,
    window_size: int = 96,
    persistence_range: int = 1,
    interval_width_fraction: float = 0.0,
) -> dict[str, list[float]]:
    """Predict household-level continuous profiles.

    Current implementation:
    - base_load, pv_gen: one-window flat moving average after persistence steps
    - base_load/pv_gen forecast bands: simple percent envelope around point forecast
    """

    payload: dict[str, list[float]] = {}
    payload.update(
        predict_base_load(
            household,
            horizon,
            window_size=window_size,
            persistence_range=persistence_range,
            interval_width_fraction=interval_width_fraction,
        )
    )
    payload.update(
        predict_pv_gen(
            household,
            horizon,
            window_size=window_size,
            persistence_range=persistence_range,
            interval_width_fraction=interval_width_fraction,
        )
    )
    return payload
