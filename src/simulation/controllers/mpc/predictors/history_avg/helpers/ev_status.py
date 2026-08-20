from __future__ import annotations

from src.runtime_config import RuntimeConfig
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

    allowed_commute_windows = RuntimeConfig.EV_COMMUTE_WINDOWS_ALLOWED[ev_key]

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
        ) -> tuple[list[int], list[int]]:
    """
    Predicts the status of a given EV (at_home, at_charging_station) for the given household and horizon.
    Uses list of state strings internally and translates to binary home/station profiles.
    """
    current_timestep = household.current_timestep
    allowed_commute_windows = RuntimeConfig.EV_COMMUTE_WINDOWS_ALLOWED[ev_key]

    end_1_latest = int(allowed_commute_windows[0]["latest_end"])
    max_commute_time_1 = allowed_commute_windows[0]["max_unavailable_steps"]

    end_2_latest = int(allowed_commute_windows[1]["latest_end"])
    max_commute_time_2 = allowed_commute_windows[1]["max_unavailable_steps"]

    # Initialize forecasts with worst case assumption
    start_1_pred, end_1_pred, start_2_pred, end_2_pred = _worst_case_commute_times(household, ev_key)

    # get status history and current status
    ev_at_home_now = getattr(household, f"{ev_key}_at_home")
    ev_at_station_now = getattr(household, f"{ev_key}_at_charging_station")

    # override worst-case predictions with observed state transitions if available
    observed_state_transitions = getattr(household, f"{ev_key}_state_transitions")
    start_1_pred = observed_state_transitions["start1"] or start_1_pred
    end_1_pred = observed_state_transitions["end1"] or end_1_pred
    start_2_pred = observed_state_transitions["start2"] or start_2_pred
    end_2_pred = observed_state_transitions["end2"] or end_2_pred

    # update prediction based on history and current status
    if not observed_state_transitions["start1"]:
        if ev_at_home_now:
            # ev is still at home
            if current_timestep < start_1_pred: # like predicted
                pass
            else:
                # ev stays home longer than predicted, move prediction to next step from now
                # and update end 1 pred to respective worst case
                start_1_pred = current_timestep + 1
                end_1_pred = min(end_1_latest, start_1_pred + max_commute_time_1 - 1)
    elif not observed_state_transitions["end1"]:
        end_1_pred = min(end_1_latest, start_1_pred + max_commute_time_1 - 1)
    elif not observed_state_transitions["start2"]:
        if ev_at_station_now:
            # ev is still at station
            if current_timestep < start_2_pred: # like predicted
                pass
            else: # ev stays at station longer than predicted, move prediction to next step from now
                # and update end 2 pred to respective worst case
                start_2_pred = current_timestep + 1
                end_2_pred = min(end_2_latest, start_2_pred + max_commute_time_2 - 1)
    elif not observed_state_transitions["end2"]:
        end_2_pred = min(end_2_latest, start_2_pred + max_commute_time_2 - 1)

    # Generate predictions
    at_home_pred = []
    at_station_pred = []
    for t in range(current_timestep, current_timestep + horizon):
        if t < start_1_pred:
            at_home_pred.append(1)
            at_station_pred.append(0)
        elif start_1_pred <= t <= end_1_pred:
            at_home_pred.append(0)
            at_station_pred.append(0)
        elif end_1_pred < t < start_2_pred:
            at_home_pred.append(0)
            at_station_pred.append(1)
        elif start_2_pred <= t <= end_2_pred:
            at_home_pred.append(0)
            at_station_pred.append(0)
        else: # pad home to the right indefinetly. this only works because we dont care about day 2
            at_home_pred.append(1)
            at_station_pred.append(0)
        
    return at_home_pred, at_station_pred
    

def predict_ev_status(
    household: Household,
    horizon: int,
    ev_key: str | None=None
) -> dict[str, list[int]]:
    '''Predicts ev1 and ev2 status (at_home, at_charging_station) for the given household, horizon'''

    if ev_key in ["ev1", "ev2"]:
        ev_home, ev_station = predict_single_ev_status(household, ev_key, horizon)
        return {
            f"{ev_key}_at_home": ev_home,
                f"{ev_key}_at_charging_station": ev_station
        }
    elif ev_key is not None:
        raise ValueError("wrong ev_key, expected ev1 or ev2")
    
    ev1_at_home, ev1_at_charging_station = predict_single_ev_status(household, "ev1", horizon)
    ev2_at_home, ev2_at_charging_station = predict_single_ev_status(household, "ev2", horizon)

    return {
        "ev1_at_home": ev1_at_home,
        "ev1_at_charging_station": ev1_at_charging_station,
        "ev2_at_home": ev2_at_home,
        "ev2_at_charging_station": ev2_at_charging_station,
    }
