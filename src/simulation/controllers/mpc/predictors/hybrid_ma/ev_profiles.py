from __future__ import annotations
from src.simulation.household import Household

# Predictors for EV-related profiles: status, driving load, and max charge.
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


def _status_value(ev_status: dict[str, list[float]], key: str, idx: int) -> float:
    series = ev_status.get(key, [])
    if 0 <= int(idx) < len(series):
        value = float(series[idx])
        return value
    return 0.0


def _exclusive_location(home_value: float, station_value: float) -> tuple[float, float]:
    home = min(1.0, max(0.0, float(home_value)))
    station = min(1.0, max(0.0, float(station_value)))
    if home <= 0.0 and station <= 0.0:
        return 0.0, 0.0
    if home >= station:
        return 1.0, 0.0
    return 0.0, 1.0


def predict_ev_load(
    household: Household,
    horizon: int,
    ev_status: dict[str, list[float]],
) -> dict[str, list[float]]:
    """Predict EV driving load as average non-zero historical load masked by status.

    Placeholder rule:
    - estimate a single expected driving load per EV from non-zero history
    - apply it only on timesteps that are neither at home nor at station
    """

    ev1_default = float(household.ev1_default_drive_load) if float(household.ev1_default_drive_load) > 0.0 else 0.0
    ev2_default = float(household.ev2_default_drive_load) if float(household.ev2_default_drive_load) > 0.0 else 0.0

    ev1_avg_drive_load = _average_non_zero(_history_values(household, "ev1_load"), default=ev1_default)
    ev2_avg_drive_load = _average_non_zero(_history_values(household, "ev2_load"), default=ev2_default)

    ev1_series: list[float] = []
    ev2_series: list[float] = []

    for i in range(max(0, int(horizon))):
        ev1_home = _status_value(ev_status, "ev1_at_home", i)
        ev1_station = _status_value(ev_status, "ev1_at_charging_station", i)
        ev2_home = _status_value(ev_status, "ev2_at_home", i)
        ev2_station = _status_value(ev_status, "ev2_at_charging_station", i)

        ev1_available = max(ev1_home, ev1_station)
        ev2_available = max(ev2_home, ev2_station)

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
) -> dict[str, list[float]]:
    # Backward-compatible alias.
    return predict_ev_load(
        household,
        horizon,
        ev_status,
    )


def predict_ev_max_charge(
    household: Household,
    horizon: int,
    ev_status: dict[str, list[float]],
) -> dict[str, list[float]]:
    """Predict EV max charge from home/station capacities masked by predicted location."""

    ev1_home_cap = float(household.ev1_max_home_charge)
    ev2_home_cap = float(household.ev2_max_home_charge)
    ev1_station_cap = float(household.ev1_max_station_charge)
    ev2_station_cap = float(household.ev2_max_station_charge)

    ev1_max: list[float] = []
    ev2_max: list[float] = []
    for i in range(horizon):
        ev1_home, ev1_station = _exclusive_location(
            _status_value(ev_status, "ev1_at_home", i),
            _status_value(ev_status, "ev1_at_charging_station", i),
        )
        ev2_home, ev2_station = _exclusive_location(
            _status_value(ev_status, "ev2_at_home", i),
            _status_value(ev_status, "ev2_at_charging_station", i),
        )

        ev1_max.append(ev1_home_cap * ev1_home + ev1_station_cap * ev1_station)
        ev2_max.append(ev2_home_cap * ev2_home + ev2_station_cap * ev2_station)

    return {
        "ev1_max_charge": ev1_max,
        "ev2_max_charge": ev2_max,
    }
