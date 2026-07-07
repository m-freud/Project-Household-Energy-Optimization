from __future__ import annotations

from src.config import Config
from src.simulation.household import Household


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


def _observed_ev_states(
    household: Household,
    home_key: str,
    station_key: str,
) -> list[str]:
    observed_steps = max(0, int(household.current_timestep))
    home_profile = household.oracle_profiles.get(home_key, [])
    station_profile = household.oracle_profiles.get(station_key, [])

    states: list[str] = []
    for idx in range(observed_steps):
        at_home = float(home_profile[idx]) > 0.0 if idx < len(home_profile) else False
        at_station = float(station_profile[idx]) > 0.0 if idx < len(station_profile) else False
        if at_home:
            states.append("home")
        elif at_station:
            states.append("station")
        else:
            states.append("driving")
    return states


def _first_state_period(states: list[str], target: str, start_period: int = 1, end_period: int | None = None) -> int | None:
    upper = len(states) if end_period is None else min(len(states), int(end_period))
    lower = max(1, int(start_period))
    for period in range(lower, upper + 1):
        if states[period - 1] == target:
            return period
    return None


def _first_non_state_period_after(states: list[str], target: str, start_period: int, end_period: int | None = None) -> int | None:
    upper = len(states) if end_period is None else min(len(states), int(end_period))
    for period in range(max(1, int(start_period)), upper + 1):
        if states[period - 1] != target:
            return period
    return None


def _predict_single_ev_status(
    household: Household,
    horizon: int,
    ev_name: str,
    home_key: str,
    station_key: str,
) -> tuple[list[float], list[float]]:
    windows = Config.EV_UNAVAILABLE_WINDOWS[ev_name]
    outbound_window = windows[0]
    return_window = windows[1]

    current_period = int(household.current_timestep)
    midpoint_period = 48
    last_period = 96

    outbound_start = int(outbound_window["earliest_start"])
    outbound_duration = max(1, int(outbound_window["max_unavailable_steps"]))
    outbound_end = outbound_start + outbound_duration - 1

    return_duration = max(1, int(return_window["max_unavailable_steps"]))
    return_end = int(return_window["latest_end"])
    return_start = return_end - return_duration + 1

    observed_states = _observed_ev_states(household, home_key, station_key)

    observed_outbound_start = _first_state_period(
        observed_states,
        "driving",
        start_period=1,
        end_period=midpoint_period,
    )

    if observed_outbound_start is not None:
        outbound_start = observed_outbound_start
        station_start = _first_state_period(
            observed_states,
            "station",
            start_period=observed_outbound_start,
            end_period=midpoint_period,
        )
        if station_start is not None:
            outbound_end = station_start - 1
            outbound_duration = max(1, outbound_end - outbound_start + 1)
        else:
            observed_duration = max(1, current_period - outbound_start + 1)
            outbound_duration = max(outbound_duration, observed_duration)
            outbound_end = outbound_start + outbound_duration - 1
    elif current_period >= outbound_start:
        # Outbound trip has not started when expected; shift the driving window right.
        outbound_start = min(last_period, current_period + 1)
        outbound_end = min(last_period, outbound_start + outbound_duration - 1)

    if current_period < midpoint_period:
        # Before noon, mirror the outbound trip around the day midpoint.
        return_duration = outbound_duration
        return_end = max(1, (last_period + 1) - outbound_start)
        return_start = max(1, return_end - return_duration + 1)

    observed_return_start = _first_state_period(
        observed_states,
        "driving",
        start_period=midpoint_period + 1,
        end_period=last_period,
    )
    if observed_return_start is not None:
        return_start = observed_return_start
        home_start = _first_state_period(
            observed_states,
            "home",
            start_period=observed_return_start,
            end_period=last_period,
        )
        if home_start is not None:
            return_end = home_start - 1
            return_duration = max(1, return_end - return_start + 1)
        else:
            observed_duration = max(1, current_period - return_start + 1)
            return_duration = max(return_duration, observed_duration)
            return_end = min(last_period, return_start + return_duration - 1)
    elif current_period >= return_start:
        current_home = bool(getattr(household, home_key))
        current_station = bool(getattr(household, station_key))
        if not current_home and not current_station:
            return_start = min(last_period, current_period + 1)
            return_end = min(last_period, return_start + return_duration - 1)

    at_home: list[float] = []
    at_station: list[float] = []
    for offset in range(max(0, int(horizon))):
        period = current_period + offset + 1
        if period < outbound_start:
            state = "home"
        elif outbound_start <= period <= outbound_end:
            state = "driving"
        elif period < return_start:
            state = "station"
        elif return_start <= period <= return_end:
            state = "driving"
        else:
            state = "home"

        at_home.append(1.0 if state == "home" else 0.0)
        at_station.append(1.0 if state == "station" else 0.0)

    return at_home, at_station


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


def predict_ev_status(household: Household, horizon: int) -> dict[str, list[float]]:
    """Predict EV status from a base commute curve plus observed updates.

    Shape assumption:
    home -> driving -> station -> driving -> home

    Base curve:
    - early outbound departure
    - long outbound drive
    - late return home
    - long return drive

    Online update rule:
    - shift or lock the outbound drive window based on observed starts/non-starts
    - before noon, mirror outbound timing/duration into the return drive
    - after noon, update the return drive from observed starts/non-starts
    """

    ev1_home, ev1_station = _predict_single_ev_status(
        household,
        horizon,
        ev_name="ev1",
        home_key="ev1_at_home",
        station_key="ev1_at_charging_station",
    )
    ev2_home, ev2_station = _predict_single_ev_status(
        household,
        horizon,
        ev_name="ev2",
        home_key="ev2_at_home",
        station_key="ev2_at_charging_station",
    )

    return {
        "ev1_at_home": ev1_home,
        "ev2_at_home": ev2_home,
        "ev1_at_charging_station": ev1_station,
        "ev2_at_charging_station": ev2_station,
    }


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
