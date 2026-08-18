from __future__ import annotations

from runtime_config import RuntimeConfig
from src.simulation.household import Household


def _observed_past_ev_states(
    household: Household,
    at_home_key: str,
    at_station_key: str,
) -> list[str]:
    home_profile = household.history.get(at_home_key, {})
    station_profile = household.history.get(at_station_key, {})

    states: list[str] = []
    observed_past_steps = int(household.current_timestep) - 1
    for period in range(1, observed_past_steps + 1):
        at_home = float(home_profile.get(period, 0.0)) > 0.0
        at_station = float(station_profile.get(period, 0.0)) > 0.0
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


def _profile_value_at_period(profile: list[float], period: int) -> float:
    # convert timestep from config to list index (0-indexed)
    idx = int(period) - 1
    if 0 <= idx < len(profile):
        return float(profile[idx])
    else:
        raise IndexError(f"Period {period} is out of bounds for profile of length {len(profile)}")


def _home_buy_price_profile(household: Household) -> list[float]:
    day_profile = getattr(household, "buy_price_day_profile", [])
    return [float(value) for value in day_profile]


def _station_buy_price_constant(household: Household, ev_name: str) -> float:
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
    # assume worst case scenario in terms of cost
    # -> charge during cheap windows before it's too late
    commute_windows = RuntimeConfig.EV_COMMUTE_WINDOWS_ALLOWED[ev_name]
    first_window = commute_windows[0]
    second_window = commute_windows[1]

    # determine earliest and latest possible start times to choose from
    earliest_first_start = int(first_window["earliest_start"])
    latest_first_start = int(first_window["latest_end"]) - commute_steps + 1
    earliest_second_start = int(second_window["earliest_start"])
    latest_second_start = int(second_window["latest_end"]) - commute_steps + 1

    # get prices
    home_prices: list[float] = _home_buy_price_profile(household) # full day profile, 0-indexed
    station_price: float = _station_buy_price_constant(household, ev_name) # constant price at station

    # Pick first commute start independently.
    # Costed over: start of day -> second window earliest start - 1
    # State model: home -> driving -> station
    # compute trajectory with highest cost sum -> same as tr. with smallest cheap window
    first_eval_end = max(1, earliest_second_start - 1)
    worst_first_start = earliest_first_start
    worst_first_cost = float("-inf")
    for first_start in range(earliest_first_start, latest_first_start + 1):
        first_end = first_start + commute_steps - 1
        total_cost = 0.0
        for period in range(1, first_eval_end + 1):
            if period < first_start:
                # charging at home before commute
                total_cost += _profile_value_at_period(home_prices, period)
            if first_start <= period <= first_end:
                # no charging during commute
                continue
            else:
                # charging at station after commute 
                total_cost += station_price

        if total_cost > worst_first_cost or (total_cost == worst_first_cost and first_start > worst_first_start):
            worst_first_cost = total_cost
            worst_first_start = first_start

    # Pick second commute start independently.
    # Costed over: first window latest end + 1 -> end of day
    # State model: station -> driving -> home
    second_eval_start = min(day_end_period, int(first_window["latest_end"]) + 1)
    best_second_start = earliest_second_start
    best_second_cost = float("-inf")
    for second_start in range(earliest_second_start, latest_second_start + 1):
        second_end = second_start + commute_steps - 1
        total_cost = 0.0
        for period in range(second_eval_start, int(day_end_period) + 1):
            if period < second_start:
                # still at station
                total_cost += station_price
            if second_start <= period <= second_end:
                # no charging during commute
                continue
            else:
                # back home after commute
                total_cost += _profile_value_at_period(home_prices, period)

        if total_cost > best_second_cost or (total_cost == best_second_cost and second_start > best_second_start):
            best_second_cost = total_cost
            best_second_start = second_start

    return int(worst_first_start), int(best_second_start)


def _predict_single_ev_status(
    household: Household,
    horizon: int,
    ev_name: str,
    home_key: str,
    station_key: str,
    commute_steps: int = 5,
) -> tuple[list[float], list[float]]:
    current_timestep = int(household.current_timestep)
    day_end_period = 96

    # base commute window assumption (worst case scenario in terms of price)
    first_start, second_start = _worst_case_two_commute_starts(
        household=household,
        ev_name=ev_name,
        commute_steps=commute_steps,
        day_end_period=day_end_period,
    )
    first_end = first_start + commute_steps - 1
    second_end = second_start + commute_steps - 1

    # get observed states to adjust windows if necessary (e.g., if commute has already started)
    observed_states = _observed_past_ev_states(household, home_key, station_key)
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
        # first start in the past
        first_start = observed_first_start
    elif state_now == "driving" and current_timestep < second_start:
        # first start is now
        first_start = current_timestep
    elif state_now == "home" and current_timestep >= first_start:
        # first start is later than predicted, shift to the right
        first_start += 1
    if observed_first_end is not None:
        first_end = max(first_start, observed_first_end)
    else:
        first_end = first_start + commute_steps - 1

    if second_start <= first_end:
        second_start = first_end + 1
    second_start = min(second_start, day_end_period)

    if observed_second_start is not None:
        # second start in the past
        second_start = observed_second_start
    elif state_now == "driving" and current_timestep >= 48:
        # no second start in the past but now driving -> second start is now
        second_start = current_timestep
    elif state_now == "station" and current_timestep >= second_start:
        # second start is late, shift right
        second_start += 1
    if observed_second_end is not None:
        second_end = max(second_start, observed_second_end)
    else:
        second_end = second_start + commute_steps - 1

    at_home: list[float] = []
    at_station: list[float] = []

    for offset in range(horizon): # prediction length is horizon, starting at current timestep
        period = current_timestep + offset
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
