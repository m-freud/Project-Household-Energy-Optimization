from __future__ import annotations

from src.config import Config
from src.simulation.household import Household


def _observed_ev_states(
    household: Household,
    at_home_key: str,
    at_station_key: str,
) -> list[str]:
    observed_steps = max(0, int(household.current_timestep))
    home_profile = household.oracle_profiles.get(at_home_key, [])
    station_profile = household.oracle_profiles.get(at_station_key, [])

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


def _profile_value_at_period(profile: list[float], period: int, default: float) -> float:
    idx = int(period) - 1
    if 0 <= idx < len(profile):
        return float(profile[idx])
    return float(default)


def _home_buy_price_profile(household: Household) -> list[float]:
    day_profile = getattr(household, "buy_price_day_profile", [])
    return [float(value) for value in day_profile]


def _constant_station_buy_price(household: Household, ev_name: str) -> float:
    attr = f"{ev_name}_station_buy_price"
    return float(getattr(household, attr, household.buy_price))


def _current_state(household: Household, home_key: str, station_key: str) -> str:
    at_home = bool(getattr(household, home_key))
    at_station = bool(getattr(household, station_key))
    if at_home:
        return "home"
    if at_station:
        return "station"
    return "driving"


def _worst_case_two_commute_starts(
    household: Household,
    ev_name: str,
    commute_steps: int,
    day_end_period: int,
) -> tuple[int, int]:
    windows = Config.EV_UNAVAILABLE_WINDOWS_ALLOWED[ev_name]
    first_window = windows[0]
    second_window = windows[1]

    first_earliest = int(first_window["earliest_start"])
    first_latest_start = int(first_window["latest_end"]) - commute_steps + 1
    second_earliest = int(second_window["earliest_start"])
    second_latest_start = int(second_window["latest_end"]) - commute_steps + 1

    # defensive programming in action
    first_latest_start = max(first_earliest, first_latest_start)
    second_latest_start = max(second_earliest, second_latest_start)

    home_prices = _home_buy_price_profile(household)
    station_price = _constant_station_buy_price(household, ev_name)
    default_home_price = float(household.buy_price)

    # Pick first commute start independently.
    # Costed over: start of day -> second window earliest start - 1
    # State model: home -> driving -> station
    first_eval_end = max(1, second_earliest - 1)
    best_first_start = first_earliest
    best_first_cost = float("-inf")
    for first_start in range(first_earliest, first_latest_start + 1):
        first_end = first_start + commute_steps - 1
        total_cost = 0.0
        for period in range(1, first_eval_end + 1):
            if first_start <= period <= first_end:
                continue
            if period < first_start:
                total_cost += _profile_value_at_period(home_prices, period, default_home_price)
            else:
                total_cost += station_price

        if total_cost > best_first_cost or (total_cost == best_first_cost and first_start > best_first_start):
            best_first_cost = total_cost
            best_first_start = first_start

    # Pick second commute start independently.
    # Costed over: first window latest end + 1 -> end of day
    # State model: station -> driving -> home
    second_eval_start = min(day_end_period, int(first_window["latest_end"]) + 1)
    best_second_start = second_earliest
    best_second_cost = float("-inf")
    for second_start in range(second_earliest, second_latest_start + 1):
        second_end = second_start + commute_steps - 1
        total_cost = 0.0
        for period in range(second_eval_start, int(day_end_period) + 1):
            if second_start <= period <= second_end:
                continue
            if period < second_start:
                total_cost += station_price
            else:
                total_cost += _profile_value_at_period(home_prices, period, default_home_price)

        if total_cost > best_second_cost or (total_cost == best_second_cost and second_start > best_second_start):
            best_second_cost = total_cost
            best_second_start = second_start

    return int(best_first_start), int(best_second_start)


def _predict_single_ev_status(
    household: Household,
    horizon: int,
    ev_name: str,
    home_key: str,
    station_key: str,
) -> tuple[list[float], list[float]]:
    current_period = int(household.current_timestep)
    day_end_period = 96
    commute_steps = 5

    first_start, second_start = _worst_case_two_commute_starts(
        household=household,
        ev_name=ev_name,
        commute_steps=commute_steps,
        day_end_period=day_end_period,
    )
    first_end = first_start + commute_steps - 1
    second_end = second_start + commute_steps - 1

    observed_states = _observed_ev_states(household, home_key, station_key)
    observed_first_start = _first_state_period(
        observed_states,
        "driving",
        start_period=1,
        end_period=48,
    )
    observed_second_start = _first_state_period(
        observed_states,
        "driving",
        start_period=49,
        end_period=day_end_period,
    )

    observed_first_end: int | None = None
    if observed_first_start is not None:
        first_non_driving = _first_non_state_period_after(
            observed_states,
            "driving",
            start_period=observed_first_start,
            end_period=48,
        )
        if first_non_driving is not None:
            observed_first_end = first_non_driving - 1

    observed_second_end: int | None = None
    if observed_second_start is not None:
        second_non_driving = _first_non_state_period_after(
            observed_states,
            "driving",
            start_period=observed_second_start,
            end_period=day_end_period,
        )
        if second_non_driving is not None:
            observed_second_end = second_non_driving - 1

    state_now = _current_state(household, home_key, station_key)

    if observed_first_start is not None:
        first_start = observed_first_start
    elif state_now == "driving" and current_period < second_start:
        # Forecast starts at current_period + 1, so align commute start to the next predicted period.
        first_start = min(day_end_period, current_period + 1)
    elif state_now == "home" and current_period >= first_start:
        # Commute should have started by now but has not; shift the first window right.
        first_start = min(day_end_period, first_start + 1)
    if observed_first_end is not None:
        first_end = max(first_start, observed_first_end)
    else:
        first_end = first_start + commute_steps - 1

    if second_start <= first_end:
        second_start = first_end + 1
    second_start = min(second_start, day_end_period)

    if observed_second_start is not None:
        second_start = observed_second_start
    elif state_now == "driving" and current_period >= 48:
        second_start = max(second_start, min(day_end_period, current_period + 1))
    elif state_now == "station" and current_period >= second_start:
        # Return commute should have started but EV is still at station; shift second window right.
        second_start = min(day_end_period, second_start + 1)
    if observed_second_end is not None:
        second_end = max(second_start, observed_second_end)
    else:
        second_end = second_start + commute_steps - 1

    at_home: list[float] = []
    at_station: list[float] = []
    for offset in range(max(0, int(horizon))):
        period = current_period + offset + 1
        if period < first_start:
            state = "home"
        elif first_start <= period <= first_end:
            state = "driving"
        elif period < second_start:
            state = "station"
        elif second_start <= period <= second_end:
            state = "driving"
        else:
            state = "home"

        at_home.append(1.0 if state == "home" else 0.0)
        at_station.append(1.0 if state == "station" else 0.0)

    return at_home, at_station


def predict_ev_status(household: Household, horizon: int) -> dict[str, list[float]]:
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
