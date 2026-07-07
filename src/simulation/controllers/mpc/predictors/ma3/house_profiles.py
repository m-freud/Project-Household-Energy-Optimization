from __future__ import annotations

from src.simulation.household import Household


def _history_values(household: Household, key: str) -> list[float]:
    history = household.history.get(key, {})
    if not history:
        return []
    return [float(history[timestep]) for timestep in sorted(history)]


def _seed_series(values: list[float], long_window: int, default: float) -> list[float]:
    seed = values[-long_window:]
    if len(seed) >= long_window:
        return seed

    needed = long_window - len(seed)
    if seed:
        return [float(seed[0])] * needed + seed
    return [float(default)] * long_window


def _forecast_moving_average(
    values: list[float],
    horizon: int,
    short_window: int,
    long_window: int,
    short_weight: float,
    default: float,
) -> list[float]:
    if horizon <= 0:
        return []

    series = _seed_series(values, long_window, default)
    forecast: list[float] = []
    long_weight = 1.0 - short_weight

    for _ in range(horizon):
        short_slice = series[-short_window:]
        long_slice = series[-long_window:]

        short_avg = sum(short_slice) / len(short_slice) if short_slice else float(default)
        long_avg = sum(long_slice) / len(long_slice) if long_slice else float(default)
        predicted = short_weight * short_avg + long_weight * long_avg

        forecast.append(float(predicted))
        series.append(float(predicted))

    return forecast


def _make_band(series: list[float], width_fraction: float) -> tuple[list[float], list[float]]:
    width_fraction = max(0.0, float(width_fraction))
    lower = [max(0.0, value * (1.0 - width_fraction)) for value in series]
    upper = [max(0.0, value * (1.0 + width_fraction)) for value in series]
    return lower, upper


def predict_base_load(
    household: Household,
    horizon: int,
    short_window: int = 7,
    long_window: int = 48,
    short_weight: float = 0.7,
    interval_width_fraction: float = 0.1,
) -> dict[str, list[float]]:
    short_window = max(1, int(short_window))
    long_window = max(short_window, int(long_window))
    short_weight = min(1.0, max(0.0, float(short_weight)))

    base_series = _forecast_moving_average(
        values=_history_values(household, "base_load"),
        horizon=horizon,
        short_window=short_window,
        long_window=long_window,
        short_weight=short_weight,
        default=float(household.base_load),
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
) -> dict[str, list[float]]:
    short_window = max(1, int(short_window))
    long_window = max(short_window, int(long_window))
    short_weight = min(1.0, max(0.0, float(short_weight)))

    pv_series = _forecast_moving_average(
        values=_history_values(household, "pv_gen"),
        horizon=horizon,
        short_window=short_window,
        long_window=long_window,
        short_weight=short_weight,
        default=float(household.pv_gen),
    )
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
        )
    )
    return payload
