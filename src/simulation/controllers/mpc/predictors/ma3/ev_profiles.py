from __future__ import annotations

from src.simulation.controllers.mpc.predictors.ma3.ev_status import predict_ev_status
from src.simulation.household import Household

# Predictors for ev status, ev_load, ev_max_charge
# (messy)

def _oracle_slice(household: Household, key: str, horizon: int) -> list[float]:
    start = int(household.current_timestep)
    profile = household.oracle_profiles.get(key, [])
    return [float(value) for value in profile[start : start + horizon]]


def _history_values(household: Household, key: str) -> list[float]:
    history = household.history.get(key, {})
    if not history:
        return []
    return [float(history[timestep]) for timestep in sorted(history)]


def _average_non_zero(values: list[float], default: float = 0.0) -> float:
    non_zero = [float(value) for value in values if float(value) > 0.0]
    if non_zero:
        return float(sum(non_zero) / len(non_zero))
    return float(default)


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

    short_window = max(1, int(short_window))
    long_window = max(short_window, int(long_window))
    short_weight = min(1.0, max(0.0, float(short_weight)))

    series = values[-long_window:]
    if len(series) < long_window:
        needed = long_window - len(series)
        if series:
            series = [float(series[0])] * needed + series
        else:
            series = [float(default)] * long_window

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


def predict_ev_load(
    household: Household,
    horizon: int,
    ev_status: dict[str, list[float]],
    short_window: int = 7,
    long_window: int = 48,
    short_weight: float = 0.7,
) -> dict[str, list[float]]:
    """Predict EV driving load as average non-zero historical load masked by status.

    Placeholder rule:
    - estimate a single expected driving load per EV from non-zero history
    - apply it only on timesteps that are neither at home nor at station
    """

    ev1_default = float(household.ev1_load) if float(household.ev1_load) > 0.0 else 0.0
    ev2_default = float(household.ev2_load) if float(household.ev2_load) > 0.0 else 0.0

    ev1_avg_drive_load = _average_non_zero(_history_values(household, "ev1_load"), default=ev1_default)
    ev2_avg_drive_load = _average_non_zero(_history_values(household, "ev2_load"), default=ev2_default)

    ev1_series: list[float] = []
    ev2_series: list[float] = []
    for i in range(max(0, int(horizon))):
        ev1_available = max(ev_status["ev1_at_home"][i], ev_status["ev1_at_charging_station"][i])
        ev2_available = max(ev_status["ev2_at_home"][i], ev_status["ev2_at_charging_station"][i])

        ev1_driving = 1.0 - ev1_available
        ev2_driving = 1.0 - ev2_available

        ev1_series.append(float(ev1_avg_drive_load) * float(max(0.0, ev1_driving)))
        ev2_series.append(float(ev2_avg_drive_load) * float(max(0.0, ev2_driving)))

    return {
        "ev1_load": ev1_series,
        "ev2_load": ev2_series,
    }


def predict_ev_loads(
    household: Household,
    horizon: int,
    ev_status: dict[str, list[float]],
    short_window: int = 7,
    long_window: int = 48,
    short_weight: float = 0.7,
) -> dict[str, list[float]]:
    # Backward-compatible alias.
    return predict_ev_load(
        household,
        horizon,
        ev_status,
        short_window=short_window,
        long_window=long_window,
        short_weight=short_weight,
    )


def predict_ev_max_charge(
    household: Household,
    horizon: int,
    ev_status: dict[str, list[float]],
) -> dict[str, list[float]]:
    """Placeholder EV max-charge predictor conditioned on predicted EV availability."""

    ev1_profile_max = _oracle_slice(
        household,
        "ev1_max_charge",
        horizon,
    )
    ev2_profile_max = _oracle_slice(
        household,
        "ev2_max_charge",
        horizon,
    )

    ev1_status = [
        max(ev_status["ev1_at_home"][i], ev_status["ev1_at_charging_station"][i])
        for i in range(horizon)
    ]
    ev2_status = [
        max(ev_status["ev2_at_home"][i], ev_status["ev2_at_charging_station"][i])
        for i in range(horizon)
    ]

    ev1_max = [ev1_profile_max[i] * ev1_status[i] for i in range(horizon)]
    ev2_max = [ev2_profile_max[i] * ev2_status[i] for i in range(horizon)]

    return {
        "ev1_max_charge": ev1_max,
        "ev2_max_charge": ev2_max,
    }
