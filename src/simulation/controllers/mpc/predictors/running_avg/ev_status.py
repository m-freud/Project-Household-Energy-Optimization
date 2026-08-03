from __future__ import annotations

from src.config import Config
from src.simulation.household import Household


def _worst_case_commute_times(
    household: Household,
    ev_key: str,
) -> tuple[int, int, int, int]:
    """
    Returns worst-case commute times for the given EV in the household,
    in terms of price options. Maximizes expensive charging windows
    -> encourages charging during cheap windows before it's too late
    start_1 : start time (first timestep of commute 1)
    end_1 : end time (last timestep of commute 1)
    start_2 : start time (first timestep of second commute window)
    end_2 : end time (last timestep of second commute window)
    """
    start_1, end_1, start_2, end_2 = 0, 0, 0, 0

    allowed_commute_windows = Config.EV_COMMUTE_WINDOWS_ALLOWED[ev_key]

    start_1_earliest = int(allowed_commute_windows[0]["earliest_start"])
    end_1_latest = int(allowed_commute_windows[0]["latest_end"])
    max_commute_time_1 = allowed_commute_windows[0]["max_unavailable_steps"]

    start_2_earliest = int(allowed_commute_windows[1]["earliest_start"])
    end_2_latest = int(allowed_commute_windows[1]["latest_end"])
    max_commute_time_2 = allowed_commute_windows[1]["max_unavailable_steps"]

    # get prices
    home_buy_price_profile = household.buy_price_day_profile # 0-indexed
    station_buy_price = getattr(household, f"{ev_key}_station_buy_price")

    # find windows that maximize expensive charging times
    eval_1_end = end_1_latest
    eval_2_start = start_2_earliest

    commute_1_max_price_sum = 0
    commute_2_max_price_sum = 0

    # find worst-case commute times for first commute window (max price sum)
    for start_1_candidate in range(start_1_earliest, end_1_latest - (max_commute_time_1 - 1) + 1):
        end_time = start_1_candidate + (max_commute_time_1 - 1)

        # evaluate price profile for this option
        # timestamps are 1-indexed, price profile is 0-indexed
        home_price_sum = sum(home_buy_price_profile[:(start_1_candidate-1)])
        station_price_sum = station_buy_price * (eval_1_end - end_time)

        total_price_sum = home_price_sum + station_price_sum
        if total_price_sum > commute_1_max_price_sum:
            commute_1_max_price_sum = total_price_sum
            start_1 = start_1_candidate
            end_1 = end_time

    # find worst-case commute times for second commute window (max price sum)
    for start_2_candidate in range(start_2_earliest, end_2_latest - (max_commute_time_2 - 1) + 1):
        end_time = start_2_candidate + (max_commute_time_2 - 1)

        # evaluate price profile for this option
        # timestamps are 1-indexed, price profile is 0-indexed
        home_price_sum = sum(home_buy_price_profile[(end_time-1)+1:]) # steps after 2nd commute window
        station_price_sum = station_buy_price * (start_2_candidate - eval_2_start)

        total_price_sum = home_price_sum + station_price_sum
        if total_price_sum > commute_2_max_price_sum:
            commute_2_max_price_sum = total_price_sum
            start_2 = start_2_candidate
            end_2 = end_time

    return start_1, end_1, start_2, end_2


def predict_single_ev_status(
        household: Household,
        ev_key: str,
        horizon: int,
        ) -> tuple[list[float], list[float]]:
    """
    Predicts the status of a given EV (at_home, at_charging_station) for the given household and horizon.
    Uses list of state strings internally and translates to binary home/station profiles.
    """
    current_timestep = household.current_timestep
    allowed_commute_windows = Config.EV_COMMUTE_WINDOWS_ALLOWED[ev_key]

    end_1_latest = int(allowed_commute_windows[0]["latest_end"])
    max_commute_time_1 = allowed_commute_windows[0]["max_unavailable_steps"]

    start_2_earliest = int(allowed_commute_windows[1]["earliest_start"])
    end_2_latest = int(allowed_commute_windows[1]["latest_end"])
    max_commute_time_2 = allowed_commute_windows[1]["max_unavailable_steps"]


    # Initialize forecasts with worst case assumption
    start_1, end_1, start_2, end_2 = _worst_case_commute_times(household, ev_key)

    # get status history and current status
    ev_at_home_history = household.history[f"{ev_key}_at_home"]
    ev_at_station_history = household.history[f"{ev_key}_at_charging_station"]
    ev_at_home_now = getattr(household, f"{ev_key}_at_home")
    ev_at_station_now = getattr(household, f"{ev_key}_at_charging_station")

    # update prediction based on history and current status

    # Phase 1
    if current_timestep <= end_1_latest:
        if ev_at_home_now:
            # ev is still at home
            if current_timestep < start_1:
                # like predicted
                pass
            else:
                # ev stays home longer than predicted, move prediction to next step
                start_1 = current_timestep + 1
                end_1 = min(end_1_latest, start_1 + max_commute_time_1 - 1)
        elif not ev_at_home_now and not ev_at_station_now:
            # ev is driving, update start 1 to first non-home timestep in history
            start_1 = next((t for t, at_home in ev_at_home_history.items() if at_home < 1), current_timestep)
            end_1 = min(end_1_latest, start_1 + max_commute_time_1 - 1)
        elif ev_at_station_now:
            # ev is at station, update start 1 to first non-home timestep in history
            # and end 1 to last non-station timestep in history
            start_1 = next((int(t) for t, at_home in ev_at_home_history.items() if at_home < 1), start_1) # fallback doesn't matter, we are already at station
            first_station_timestep = next((int(t) for t, at_station in ev_at_station_history.items() if at_station > 0), current_timestep)
            end_1 = first_station_timestep - 1

    # Phase 2
    if current_timestep >= start_2_earliest:
        if ev_at_station_now:
            # ev is still at station
            if current_timestep < start_2:
                # like predicted
                pass
            else:
                # ev stays at station longer than predicted, move prediction to next step
                start_2 = current_timestep + 1
                end_2 = min(end_2_latest, start_2 + max_commute_time_2 - 1)
        elif not ev_at_home_now and not ev_at_station_now:
            # ev is driving, update start 2 to first non-station timestep in history
            start_2 = next((int(t) for t, at_station in ev_at_station_history.items() if t >= start_2_earliest and at_station < 1), current_timestep)
            end_2 = min(end_2_latest, start_2 + max_commute_time_2 - 1)
        elif ev_at_home_now:
            # ev is back at home, update start 2 to first non-station timestep in history
            # and end 2 to last non-home timestep in history
            start_2 = next((int(t) for t, at_station in ev_at_station_history.items() if t >= start_2_earliest and at_station < 1), start_2) # fallback doesn't matter, we are already back home
            first_home_timestep = next((int(t) for t, at_home in ev_at_home_history.items() if t >= start_2_earliest and at_home > 0), current_timestep)
            end_2 = first_home_timestep - 1
    

    # Generate predictions
    at_home_pred = []
    at_station_pred = []
    for t in range(current_timestep, current_timestep + horizon):
        if t < start_1:
            at_home_pred.append(1)
            at_station_pred.append(0)
        elif start_1 <= t <= end_1:
            at_home_pred.append(0)
            at_station_pred.append(0)
        elif end_1 < t < start_2:
            at_home_pred.append(0)
            at_station_pred.append(1)
        elif start_2 <= t <= end_2:
            at_home_pred.append(0)
            at_station_pred.append(0)
        else: # pad home to the right indefinetly. this only works because we dont care about day 2
            at_home_pred.append(1)
            at_station_pred.append(0)
        
    return at_home_pred, at_station_pred
    


def predict_ev_status(
    household: Household,
    horizon: int,
) -> dict[str, list[float]]:
    '''Predicts ev1 and ev2 status (at_home, at_charging_station) for the given household, horizon'''
    
    ev1_at_home, ev1_at_charging_station = predict_single_ev_status(household, "ev1", horizon)
    ev2_at_home, ev2_at_charging_station = predict_single_ev_status(household, "ev2", horizon)

    return {
        "ev1_at_home": ev1_at_home,
        "ev1_at_charging_station": ev1_at_charging_station,
        "ev2_at_home": ev2_at_home,
        "ev2_at_charging_station": ev2_at_charging_station,
    }