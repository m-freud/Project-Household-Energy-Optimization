from src.simulation.household import Household
from src.simulation.controllers.mpc.predictors.xgboost.encode_time_cyclic import encode_time_cyclic
from src.simulation.controllers.mpc.predictors.running_avg import (
    predict_single_ev_status as predict_ev_status_worst_case,
)

from xgboost import XGBClassifier

from src.config import Config


def _has_been_observed_ev_at_station(household: Household, ev_key: str) -> bool:
    """
    Checks if the household has been observed at the station.

    Args:
        household (Household): The household to check.
        ev_key (str): The key identifying the EV.
    """
    if getattr(household, f"{ev_key}_at_station", None):
        return True
    
    at_station_history = getattr(household, f"{ev_key}_at_station_history", None)
    if at_station_history and any(at_station_history):
        return True
    
    return False


def _fetch_ev_status_data(household: Household, ev_key: str) -> dict:
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
        "at_station_now": getattr(household, f"{ev_key}_at_station", 0),
        "at_home_now": getattr(household, f"{ev_key}_at_home", 0),

        # histories
        "at_station_history": getattr(household.history, f"{ev_key}_at_charging_station", []),
        "at_home_history": getattr(household.history, f"{ev_key}_at_home", []),

        # observed state transitions
        "start_1": ""
    }
    return ev_status_data



def _build_ev_status_features(household: Household, ev_key: str, horizon:int)-> dict:
    features = {
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
    # If the household has been observed at the station, use worst-case prediction
    if not _has_been_observed_ev_at_station(household, ev_key):
        # Phase 1 - Use worst-case prediction based like in running average predictor
        return predict_ev_status_worst_case(household, ev_key, horizon)
    else:
        # Phase 2: Use XGBoost model for prediction
        features = _build_ev_status_features(household, ev_key, horizon)
        pred = model.predict(features)
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
