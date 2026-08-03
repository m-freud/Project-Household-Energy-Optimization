def make_band(series: list[float], width_fraction: float) -> tuple[list[float], list[float]]:
    width_fraction = max(0.0, float(width_fraction))
    lower = [max(0.0, value * (1.0 - width_fraction)) for value in series]
    upper = [max(0.0, value * (1.0 + width_fraction)) for value in series]
    # The first forecast point corresponds to a known current-state value,
    # so its interval should have zero width.
    if series:
        lower[0] = float(series[0])
        upper[0] = float(series[0])
    return lower, upper
