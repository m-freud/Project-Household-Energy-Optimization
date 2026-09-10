from src.simulation.controllers.mpc.predictors.ml.model_interface import TClassifier
from src.simulation.household import Household
from src.runtime_config import RuntimeConfig
from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._ev_status import _build_ev_status_features, _try_bypass


def _predict_single_ev_status(model: TClassifier | None, household: Household, ev_key: str, horizon: int=96) -> tuple[list[int], list[int]]:
    """
    Predicts the status of a single EV (at_home, at_charging_station) for the given household and horizon.

    Args:
        household (Household): The household for which to predict EV status.
        ev_key (str): The key identifying the EV.
        horizon (int): The number of time steps to predict.

    Returns:
        tuple[list[int], list[int]]: Two lists representing the predicted status of the EV at home and at the charging station.
    """
    if model is None:
        return [0] * horizon, [0] * horizon

    def _status_to_home_station(status: int) -> tuple[int, int]:
        # prediction has shape of at_home, at_station, so we convert back for pred
        if status == 0:
            return 1, 0
        if status == 1:
            return 0, 0
        if status == 2:
            return 0, 1
        raise ValueError(f"Unexpected EV status class: {status}")

    windows = RuntimeConfig.EV_COMMUTE_WINDOWS_ALLOWED[ev_key]
    start1_earliest = windows[0]["earliest_start"]
    end1_latest = windows[0]["latest_end"]
    start2_earliest = windows[1]["earliest_start"]
    end2_latest = windows[1]["latest_end"]
    
    ev_home_key = f"{ev_key}_at_home"
    ev_station_key = f"{ev_key}_at_charging_station"

    current_at_home = int(getattr(household, ev_home_key, 0)) # bool to int
    current_at_station = int(getattr(household, ev_station_key, 0)) # bool to int
    current_status = 1 - current_at_home + current_at_station  # 012 conversion
    current_timestep = household.current_timestep

    # init sim history for prediction
    at_home_history = household.history.get(f"{ev_key}_at_home", {})
    at_station_history = household.history.get(f"{ev_key}_at_charging_station", {})

    # we use list here for convenience
    sim_status_history = [ # combine to single 012
        1 - at_home_history[step] + at_station_history[step]
        for step in range(1, current_timestep)
    ]

    at_home_pred, at_station_pred = [current_at_home], [current_at_station]

    for _ in range(horizon - 1):
        ev_status_data = {
            "timestep": current_timestep,
            "status": current_status,
            "status_history": sim_status_history,
            "start1_earliest": start1_earliest,
            "end1_latest": end1_latest,
            "start2_earliest": start2_earliest,
            "end2_latest": end2_latest,
            "max_commute_steps_1": windows[0]["max_unavailable_steps"],
            "max_commute_steps_2": windows[1]["max_unavailable_steps"],
        }

        # here we need features before bypass because bypass depends on some of the features
        all_features = _build_ev_status_features(ev_status_data)

        # bypass model if possible
        bypass = _try_bypass(current_timestep, all_features)
        if bypass is not None:
            current_status = bypass
            current_at_home, current_at_station = _status_to_home_station(current_status)
            at_home_pred.append(current_at_home)
            at_station_pred.append(current_at_station)
            current_timestep += 1
            continue

        model_family_name = ModelConfig.get_model_family_name(model)
        model_features = ModelConfig.MODEL_FEATURES_BY_FAMILY[model_family_name]["ev_status"]

        # ensure completeness of features for the model
        for f in model_features:
            if f not in all_features:
                raise ValueError(f"Missing feature '{f}' in features dictionary.")

        # ensure correct order of features for the model
        model_input = [all_features[f] for f in model_features]

        # update history before predicting next value
        sim_status_history.append(current_status)

        # predict next value, save as split into at_home and at_station
        current_status = int(model.predict([model_input])[0])  # type: ignore
        
        current_at_home, current_at_station = _status_to_home_station(current_status)
        at_home_pred.append(current_at_home)
        at_station_pred.append(current_at_station)

        # increment time
        current_timestep += 1

    return at_home_pred, at_station_pred
    

def predict_ev_status(
    model_ev1: TClassifier|None,
    model_ev2: TClassifier|None,
    household: Household,
    horizon: int,
    ev_key: str | None = None
) -> dict[str, list[int]]:
    '''Predicts ev1 and ev2 status (at_home, at_charging_station) for the given household, horizon'''

    if ev_key in ["ev1", "ev2"]:
        model = model_ev1 if ev_key == "ev1" else model_ev2
        ev_home, ev_station = _predict_single_ev_status(model, household, ev_key, horizon)
        return {
            f"{ev_key}_at_home": ev_home,
                f"{ev_key}_at_charging_station": ev_station
        }
    elif ev_key is not None:
        raise ValueError("wrong ev_key, expected ev1 or ev2")

    ev1_at_home, ev1_at_charging_station = _predict_single_ev_status(model_ev1, household, "ev1", horizon)
    ev2_at_home, ev2_at_charging_station = _predict_single_ev_status(model_ev2, household, "ev2", horizon)

    return {
        "ev1_at_home": ev1_at_home,
        "ev1_at_charging_station": ev1_at_charging_station,
        "ev2_at_home": ev2_at_home,
        "ev2_at_charging_station": ev2_at_charging_station,
    }

