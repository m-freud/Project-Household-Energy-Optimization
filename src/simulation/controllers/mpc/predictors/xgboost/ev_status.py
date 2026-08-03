
from src.simulation.household import Household

from src.simulation.controllers.mpc.predictors.running_avg import (
    predict_single_ev_status as predict_ev_status_worst_case,
)


from xgboost import XGBClassifier


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


def _get_features_for_ev_status(household: Household, ev_key: str, horizon:int)-> list[list[float]]:
    features = [[0.0] * 10 for _ in range(horizon)]  # Example: 10 features, all zeros
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
        # Phase 1
        return predict_ev_status_worst_case(household, ev_key, horizon)
    else:
        features = _get_features_for_ev_status
        pred = model.predict(features)
        return pred

    



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