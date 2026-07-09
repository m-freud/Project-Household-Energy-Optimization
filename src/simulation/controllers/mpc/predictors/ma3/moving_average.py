from __future__ import annotations

import math


def seed_series(values: list[float], long_window: int, default: float) -> list[float]:
    seed = values[-long_window:]
    if len(seed) >= long_window:
        return seed

    needed = long_window - len(seed)
    if seed:
        return [float(seed[0])] * needed + seed
    return [float(default)] * long_window


def recent_slope(values: list[float], trend_window: int) -> float:
    trend_window = max(2, int(trend_window))
    if len(values) < 2:
        return 0.0

    tail = [float(value) for value in values[-trend_window:]]
    if len(tail) < 2:
        return 0.0

    diffs = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
    if not diffs:
        return 0.0
    return float(sum(diffs) / len(diffs))


def persistence_alpha(
    step_index: int,
    mode: str,
    persistence_horizon: int,
    constant_alpha: float = 0.5,
) -> float:
    step_index = max(1, int(step_index))
    persistence_horizon = max(1, int(persistence_horizon))
    constant_alpha = min(1.0, max(0.0, float(constant_alpha)))

    if mode == "none":
        return 1.0

    if mode == "constant":
        return constant_alpha

    if step_index == 1:
        return 0.0

    if persistence_horizon == 1:
        return 1.0

    if mode == "linear":
        return min(1.0, float(step_index - 1) / float(persistence_horizon - 1))

    tau = float(persistence_horizon - 1) / math.log(10.0)
    return min(1.0, 1.0 - math.exp(-float(step_index - 1) / tau))


def forecast_moving_average(
    values: list[float],
    horizon: int,
    short_window: int,
    long_window: int,
    short_weight: float,
    default: float,
    persistence_mode: str = "exponential",
    persistence_horizon: int = 8,
    persistence_constant_alpha: float = 0.5,
    trend_weight: float = 0.0,
    trend_window: int = 4,
) -> list[float]:
    if horizon <= 0:
        return []

    series = seed_series(values, long_window, default)
    forecast: list[float] = []
    long_weight = 1.0 - short_weight
    persistence_anchor = float(default)
    slope = recent_slope(values, trend_window)

    for step_index in range(1, horizon + 1):
        short_slice = series[-short_window:]
        long_slice = series[-long_window:]

        short_avg = sum(short_slice) / len(short_slice) if short_slice else float(default)
        long_avg = sum(long_slice) / len(long_slice) if long_slice else float(default)
        ma_predicted = short_weight * short_avg + long_weight * long_avg
        alpha = persistence_alpha(
            step_index,
            persistence_mode,
            persistence_horizon,
            constant_alpha=persistence_constant_alpha,
        )
        predicted = (1.0 - alpha) * persistence_anchor + alpha * ma_predicted
        predicted = predicted + float(trend_weight) * float(step_index) * slope

        forecast.append(float(predicted))
        series.append(float(predicted))

    return forecast