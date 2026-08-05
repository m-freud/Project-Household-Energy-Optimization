from src.simulation.household import Household
from src.simulation.controllers.mpc.predictors.xgboost.encode_time_cyclic import encode_time_cyclic
from src.simulation.controllers.mpc.predictors.running_avg import (
    predict_single_ev_status as predict_ev_status_worst_case,
)

from xgboost import XGBClassifier

from src.config import Config


def _fetch_ev_status_data(household: Household, ev_key: str) -> dict:
    state_transitions = getattr(household, f"{ev_key}_state_transitions")

    ev_status_data = {
        # current time encoded cyclically -> reflect evening/morning proximity
        "current_time": encode_time_cyclic(household.current_timestep, Config.TOTAL_TIMESTEPS_DAY),

        # config
        "earliest_start_1": Config.EV_COMMUTE_WINDOWS_ALLOWED[ev_key][0]["earliest_start"],
        "latest_end_1": Config.EV_COMMUTE_WINDOWS_ALLOWED[ev_key][0]["latest_end"],
        "earliest_start_2": Config.EV_COMMUTE_WINDOWS_ALLOWED[ev_key][1]["earliest_start"],
        "latest_end_2": Config.EV_COMMUTE_WINDOWS_ALLOWED[ev_key][1]["latest_end"],
        "max_unavailable_steps_1": Config.EV_COMMUTE_WINDOWS_ALLOWED[ev_key][0]["max_unavailable_steps"],
        "max_unavailable_steps_2": Config.EV_COMMUTE_WINDOWS_ALLOWED[ev_key][1]["max_unavailable_steps"],

        # current states
        "at_station_now": getattr(household, f"{ev_key}_at_charging_station", 0),
        "at_home_now": getattr(household, f"{ev_key}_at_home", 0),

        # histories
        "at_station_history": household.history.get(f"{ev_key}_at_charging_station", {}),
        "at_home_history": household.history.get(f"{ev_key}_at_home", {}),

        # observed state transitions
        "start1": state_transitions["start1"],
        "end1": state_transitions["end1"],
        "start2": state_transitions["start2"],
        "end2": state_transitions["end2"],
    }
    
    return ev_status_data



def _build_ev_status_features(ev_status_data)-> dict:
    features = {
        "current_time_sin": ev_status_data["current_time"][0],
        "current_time_cos": ev_status_data["current_time"][1],

    }
    return features


def _predict_single_ev_status(model: XGBClassifier, household: Household, ev_key: str, horizon: int) -> tuple[list[float], list[float]]:
    """
    Predicts the status of a single EV (at_home, at_charging_station) for the given household and horizon.

    Args:
        household (Household): The household for which to predict EV status.
        ev_key (str): The key identifying the EV.
        horizon (int): The number of time steps to predict.

    Returns:
        tuple[list[float], list[float]]: Two lists representing the predicted status of the EV at home and at the charging station.
    """
    # note: it is possible to use worst-case pred until start1 but we try full xgb for now.
    ev_status_data = _fetch_ev_status_data(household, ev_key)
    features = _build_ev_status_features(ev_status_data)
    pred = model.predict(features) #TODO this is wrong we need n timesteps
    return pred[:, 0], pred[:, 1]
    


def predict_ev_status(
    model: XGBClassifier,
    household: Household,
    horizon: int,
) -> dict[str, list[float]]:
    '''Predicts ev1 and ev2 status (at_home, at_charging_station) for the given household, horizon'''
    
    ev1_at_home, ev1_at_charging_station = _predict_single_ev_status(model, household, "ev1", horizon)
    ev2_at_home, ev2_at_charging_station = _predict_single_ev_status(model, household, "ev2", horizon)

    return {
        "ev1_at_home": ev1_at_home,
        "ev1_at_charging_station": ev1_at_charging_station,
        "ev2_at_home": ev2_at_home,
        "ev2_at_charging_station": ev2_at_charging_station,
    }
