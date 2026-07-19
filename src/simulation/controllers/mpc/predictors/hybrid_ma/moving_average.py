
def forecast_history_average(
    values: list[float],
    horizon: int,
    default: float = 0.0, # default value if no history is available, most profiles start low
) -> list[float]:
    hist_avg = float(sum(values) / len(values)) if values else float(default)
    forecast = [hist_avg] * horizon
    forecast[0] = float(values[-1]) if values else float(default)
    return forecast
