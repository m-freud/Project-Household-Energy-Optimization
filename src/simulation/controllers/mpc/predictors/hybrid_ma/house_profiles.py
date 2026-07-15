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
    # The first forecast point corresponds to a known current-state value,
    # so its interval should have zero width.
    if series:
        lower[0] = float(series[0])
        upper[0] = float(series[0])
    return lower, upper


def _apply_pv_window_mask(household: Household, series: list[float]) -> list[float]:
    window = Config.PV_GENERATION_WINDOW_OBSERVED
    start_period = int(window["earliest_start"])
    end_period = int(window["latest_end"])
    current_period = int(household.current_timestep)

    masked: list[float] = []
    for idx, value in enumerate(series):
        period = current_period + idx + 1
        if start_period <= period <= end_period:
            masked.append(float(value))
        else:
            masked.append(0.0)
    return masked


def predict_base_load(
    household: Household,
    horizon: int,
    short_window: int = 7,
    long_window: int = 48,
    short_weight: float = 0.7,
    interval_width_fraction: float = 0.1,
    persistence_mode: str = "exponential",
    persistence_range: int = 8,
    persistence_constant_alpha: float = 0.5,
    trend_weight: float = 0.0,
    trend_window: int = 4,
    trend_range: int = 4,
) -> dict[str, list[float]]:
    short_window = max(0, int(short_window))
    long_window = max(max(1, short_window), int(long_window))
    short_weight = min(1.0, max(0.0, float(short_weight)))

    base_series = forecast_moving_average(
        values=_history_values(household, "base_load"),
        horizon=horizon,
        short_window=short_window,
        long_window=long_window,
        short_weight=short_weight,
        default=float(household.base_load),
        persistence_mode=persistence_mode,
        persistence_range=persistence_range,
        persistence_constant_alpha=persistence_constant_alpha,
        trend_weight=trend_weight,
        trend_window=trend_window,
        trend_range=trend_range,
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
    short_window: int = 7,
    long_window: int = 48,
    short_weight: float = 0.7,
    interval_width_fraction: float = 0.1,
    persistence_mode: str = "exponential",
    persistence_range: int = 8,
    persistence_constant_alpha: float = 0.5,
    trend_weight: float = 0.0,
    trend_window: int = 4,
    trend_range: int = 4,
) -> dict[str, list[float]]:
    short_window = max(0, int(short_window))
    long_window = max(max(1, short_window), int(long_window))
    short_weight = min(1.0, max(0.0, float(short_weight)))

    pv_series = forecast_moving_average(
        values=_history_values(household, "pv_gen"),
        horizon=horizon,
        short_window=short_window,
        long_window=long_window,
        short_weight=short_weight,
        default=float(household.pv_gen),
        persistence_mode=persistence_mode,
        persistence_range=persistence_range,
        persistence_constant_alpha=persistence_constant_alpha,
        trend_weight=trend_weight,
        trend_window=trend_window,
        trend_range=trend_range,
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
    short_window: int = 7,
    long_window: int = 48,
    short_weight: float = 0.7,
    interval_width_fraction: float = 0.1,
    persistence_mode: str = "exponential",
    persistence_range: int = 8,
    persistence_constant_alpha: float = 0.5,
    trend_weight: float = 0.0,
    trend_window: int = 4,
    trend_range: int = 4,
) -> dict[str, list[float]]:
    """Predict household-level continuous profiles.

    Placeholder MA3 implementation:
    - base_load, pv_gen: blended short/long moving average
    - base_load/pv_gen forecast bands: simple percent envelope around point forecast
    """

    payload: dict[str, list[float]] = {}
    payload.update(
        predict_base_load(
            household,
            horizon,
            short_window=short_window,
            long_window=long_window,
            short_weight=short_weight,
            interval_width_fraction=interval_width_fraction,
            persistence_mode=persistence_mode,
            persistence_range=persistence_range,
            persistence_constant_alpha=persistence_constant_alpha,
            trend_weight=trend_weight,
            trend_window=trend_window,
            trend_range=trend_range,
        )
    )
    payload.update(
        predict_pv_gen(
            household,
            horizon,
            short_window=short_window,
            long_window=long_window,
            short_weight=short_weight,
            interval_width_fraction=interval_width_fraction,
            persistence_mode=persistence_mode,
            persistence_range=persistence_range,
            persistence_constant_alpha=persistence_constant_alpha,
            trend_weight=trend_weight,
            trend_window=trend_window,
            trend_range=trend_range,
        )
    )
    return payload
