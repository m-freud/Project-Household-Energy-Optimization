from __future__ import annotations


def seed_series(values: list[float], window_size: int, default: float) -> list[float]:
    window_size = max(1, int(window_size))
    seed = values[-window_size:]
    if len(seed) >= window_size:
        return seed

    needed = window_size - len(seed)
    if seed:
        return [float(seed[0])] * needed + seed
    return [float(default)] * window_size


def forecast_moving_average(
    values: list[float],
    horizon: int,
    window_size: int,
    default: float = 0.0,
) -> list[float]:
    if horizon <= 0:
        return []

    window_size = max(1, int(window_size))
    series = seed_series(values, window_size, default)
    forecast: list[float] = []
    flat_predicted = float(sum(series[-window_size:]) / len(series[-window_size:])) if series else float(default)

    for _ in range(horizon):
        predicted = flat_predicted
        forecast.append(float(predicted))

    return forecast