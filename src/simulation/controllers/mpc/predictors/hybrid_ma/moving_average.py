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
    persistence_range: int = 1,
) -> list[float]:
    if horizon <= 0:
        return []

    window_size = max(1, int(window_size))
    persistence_range = max(0, int(persistence_range))
    series = seed_series(values, window_size, default)
    forecast: list[float] = []
    last_value = float(values[-1]) if values else float(default)

    for step_index in range(1, horizon + 1):
        if step_index <= persistence_range:
            predicted = float(last_value)
            forecast.append(predicted)
            series.append(predicted)
            continue

        window = series[-window_size:]
        predicted = sum(window) / len(window) if window else float(default)
        forecast.append(float(predicted))
        series.append(float(predicted))

    return forecast