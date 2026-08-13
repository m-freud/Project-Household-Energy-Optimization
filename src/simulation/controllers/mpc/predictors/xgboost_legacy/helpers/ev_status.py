
# paste this to enable src. imports
from pathlib import Path
import sys

import pandas as pd

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.household import Household
from src.simulation.devices.ev import EV
from src.simulation.controllers.mpc.predictors.ml.helpers.encode_time_cyclic import encode_time_cyclic

from xgboost import XGBClassifier

from src.config import Config


def _get_phase_id_seq(status_seq: list[int]) -> list[int]:
    """
    turn 012 state sequence into 01234 phase sequence
    0: at home, before first commute
    1: first commute, before first station visit
    2: first station visit
    3: second commute, before second station visit
    4: second station visit
    """
    phase_ids: list[int] = []
    seen_station = False

    for state in status_seq:
        state_i = int(state)
        if state_i == 2:
            phase_ids.append(2)
            seen_station = True
        elif state_i == 1 and not seen_station:
            phase_ids.append(1)
        elif state_i == 1 and seen_station:
            phase_ids.append(3)
        elif state_i == 0 and not seen_station:
            phase_ids.append(0)
        else:
            phase_ids.append(4)

    return phase_ids


def _get_observed_commute_boundaries(phase_ids: list[int]) -> tuple[int, int, int, int]:
    """
    given phase ids: 000111222333444
    returns the observed start and end of the first and second commute phases (1 and 3)

    strategy: cycle through and update depending on how far we got
    """
    start1 = end1 = start2 = end2 = -1

    for i, phase_id in enumerate(phase_ids):
        if phase_id == 1 and start1 == -1:
            start1 = i + 1 # convert to timestep
        if phase_id == 2 and end1 == -1:
            end1 = (i - 1) + 1 # convert to timestep 
        if phase_id == 3 and start2 == -1:
            start2 = i + 1 # convert to timestep
        if phase_id == 4 and end2 == -1:
            end2 = (i - 1) + 1 # convert to timestep

    return start1, end1, start2, end2


def _fetch_ev_status_data(household: Household, ev_key: str) -> dict:
    """
    fetch ev data used for feature building
    we mostly separate fetching from building
    status is converted to 012 format
    """
    windows = Config.EV_COMMUTE_WINDOWS_ALLOWED[ev_key]
    current_timestep = household.current_timestep

    at_home_history = household.history.get(f"{ev_key}_at_home", {})
    at_station_history = household.history.get(f"{ev_key}_at_charging_station", {})

    status_history = [
        int(1 - int(at_home_history.get(step, 0)) + int(at_station_history.get(step, 0)))
        for step in range(1, current_timestep)
    ]

    current_status = int( # home, station to 012 conversion
        1
        - int(getattr(household, f"{ev_key}_at_home", 0))
        + int(getattr(household, f"{ev_key}_at_charging_station", 0))
    )

    return {
        "timestep": int(current_timestep),
        "status": int(current_status),
        "status_history": status_history,
        "start1_earliest": int(windows[0]["earliest_start"]),
        "end1_latest": int(windows[0]["latest_end"]),
        "start2_earliest": int(windows[1]["earliest_start"]),
        "end2_latest": int(windows[1]["latest_end"]),
        "max_commute_steps_1": int(windows[0]["max_unavailable_steps"]),
        "max_commute_steps_2": int(windows[1]["max_unavailable_steps"]),
    }


def _build_ev_status_features(ev_status_data) -> dict:
    """
    build features ready to be used in classification model
    """
    timestep = ev_status_data["timestep"]
    status = ev_status_data["status"]
    status_history = list(ev_status_data.get("status_history", []))
    status_seq = status_history + [status]

    def _lag(lag: int) -> tuple[int, int]:
        idx = len(status_seq) - 1 - lag
        if idx >= 0:
            return int(status_seq[idx]), 0
        return -1, 1

    def _steps_in_current_state() -> int:
        steps = 1
        for previous in reversed(status_seq[:-1]):
            if int(previous) == status:
                steps += 1
            else:
                break
        return int(steps)

    phase_ids = _get_phase_id_seq(status_seq)

    start1, end1, start2, end2 = _get_observed_commute_boundaries(phase_ids)

    start1_observed = int(start1 != -1)
    end1_observed = int(end1 != -1)
    start2_observed = int(start2 != -1)
    end2_observed = int(end2 != -1)

    observed_window_length_1 = int(end1 - start1 + 1) if end1_observed else -1
    observed_window_length_2 = int(end2 - start2 + 1) if end2_observed else -1

    max_commute_steps_1 = int(ev_status_data["max_commute_steps_1"])
    max_commute_steps_2 = int(ev_status_data["max_commute_steps_2"])

    window_length_slack_1 = int(max_commute_steps_1 - observed_window_length_1) if observed_window_length_1 != -1 else -1
    window_length_slack_2 = int(max_commute_steps_2 - observed_window_length_2) if observed_window_length_2 != -1 else -1

    def _steps_to_boundary(boundary: int) -> int:
        if timestep <= boundary:
            return int(boundary - timestep + 1)
        return int(boundary - timestep)

    status_lag_1, status_lag_1_is_pad = _lag(1)
    status_lag_2, status_lag_2_is_pad = _lag(2)
    status_lag_4, status_lag_4_is_pad = _lag(4)
    status_lag_8, status_lag_8_is_pad = _lag(8)

    features = {
        "timestep": timestep,
        "status": status,
        "time_sin": float(encode_time_cyclic(timestep)[0]),
        "time_cos": float(encode_time_cyclic(timestep)[1]),
        "steps_in_current_state": _steps_in_current_state(),
        "phase_id": phase_ids[-1],
        "status_lag_1": status_lag_1,
        "status_lag_1_is_pad": status_lag_1_is_pad,
        "status_lag_2": status_lag_2,
        "status_lag_2_is_pad": status_lag_2_is_pad,
        "status_lag_4": status_lag_4,
        "status_lag_4_is_pad": status_lag_4_is_pad,
        "status_lag_8": status_lag_8,
        "status_lag_8_is_pad": status_lag_8_is_pad,
        "start1_earliest": ev_status_data["start1_earliest"],
        "end1_latest": ev_status_data["end1_latest"],
        "start2_earliest": ev_status_data["start2_earliest"],
        "end2_latest": ev_status_data["end2_latest"],
        "max_commute_steps_1": max_commute_steps_1,
        "max_commute_steps_2": max_commute_steps_2,
        "steps_to_start1_earliest": _steps_to_boundary(ev_status_data["start1_earliest"]),
        "steps_to_end1_latest": _steps_to_boundary(ev_status_data["end1_latest"]),
        "steps_to_start2_earliest": _steps_to_boundary(ev_status_data["start2_earliest"]),
        "steps_to_end2_latest": _steps_to_boundary(ev_status_data["end2_latest"]),
        "start1": start1,
        "end1": end1,
        "start2": start2,
        "end2": end2,
        "start1_observed": start1_observed,
        "end1_observed": end1_observed,
        "start2_observed": start2_observed,
        "end2_observed": end2_observed,
        "observed_window_length_1": observed_window_length_1,
        "observed_window_length_2": observed_window_length_2,
        "window_length_slack_1": window_length_slack_1,
        "window_length_slack_2": window_length_slack_2,
    }

    return features


def _try_bypass(current_time: int, features: dict) -> int | None:
    """
    bypass the model if we are past the last timestep
    """
    if current_time >= 96:
        return 0 # at home

    if current_time < features["start1_earliest"] or features["end2_observed"]:
        return 0 # at home

    if (features["end1_observed"] or current_time > features["end1_latest"]) and current_time < features["start2_earliest"]:
        return 2 # at station

    return None


def _predict_single_ev_status(model: XGBClassifier, household: Household, ev_key: str, horizon: int=96) -> tuple[list[int], list[int]]:
    """
    Predicts the status of a single EV (at_home, at_charging_station) for the given household and horizon.

    Args:
        household (Household): The household for which to predict EV status.
        ev_key (str): The key identifying the EV.
        horizon (int): The number of time steps to predict.

    Returns:
        tuple[list[float], list[float]]: Two lists representing the predicted status of the EV at home and at the charging station.
    """
    def _status_to_home_station(status: int) -> tuple[int, int]:
        # prediction has shape of at_home, at_station, so we convert back for pred
        if status == 0:
            return 1, 0
        if status == 1:
            return 0, 0
        if status == 2:
            return 0, 1
        raise ValueError(f"Unexpected EV status class: {status}")

    windows = Config.EV_COMMUTE_WINDOWS_ALLOWED[ev_key]
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

        features = _build_ev_status_features(ev_status_data)

        # bypass model if possible
        bypass = _try_bypass(current_timestep, features)
        if bypass is not None:
            current_status = bypass
            current_at_home, current_at_station = _status_to_home_station(current_status)
            at_home_pred.append(current_at_home)
            at_station_pred.append(current_at_station)
            current_timestep += 1
            continue

        # ensure completeness of features for the model
        for f in Config.XGB_FEATURES["EV_STATUS"]:
            if f not in features:
                raise ValueError(f"Missing feature '{f}' in features dictionary.")

        # ensure correct order of features for the model
        model_input = [features[f] for f in Config.XGB_FEATURES["EV_STATUS"]]

        # update history before predicting next value
        sim_status_history.append(current_status)

        # predict next value, save as split into at_home and at_station
        current_status = int(model.predict([model_input])[0])
        
        current_at_home, current_at_station = _status_to_home_station(current_status)
        at_home_pred.append(current_at_home)
        at_station_pred.append(current_at_station)

        # increment time
        current_timestep += 1

    return at_home_pred, at_station_pred
    

def predict_ev_status(
    model_ev1: XGBClassifier,
    model_ev2: XGBClassifier,
    household: Household,
    horizon: int,
) -> dict[str, list[int]]:
    '''Predicts ev1 and ev2 status (at_home, at_charging_station) for the given household, horizon'''

    ev1_at_home, ev1_at_charging_station = _predict_single_ev_status(model_ev1, household, "ev1", horizon)
    ev2_at_home, ev2_at_charging_station = _predict_single_ev_status(model_ev2, household, "ev2", horizon)

    return {
        "ev1_at_home": ev1_at_home,
        "ev1_at_charging_station": ev1_at_charging_station,
        "ev2_at_home": ev2_at_home,
        "ev2_at_charging_station": ev2_at_charging_station,
    }


if __name__ == "__main__":
    from training._features.ev_status_features import get_ev_status_features

    household_id = 1
    ev_key = "ev1"

    feature_df = get_ev_status_features([household_id])
    ev_rows = (
        feature_df[
            (feature_df["household_id"] == household_id)
            & (feature_df["ev_key"] == ev_key)
        ]
        .sort_values("timestep")
        .reset_index(drop=True)
    )

    training_boundary_df = ev_rows[["timestep", "status", "start1", "end1", "start2", "end2"]].copy()

    ev1 = EV(capacity=1.0, max_charge=1.0, max_discharge=1.0, efficiency=1.0, name=ev_key)
    household = Household(player_id=household_id, start_time=1, ev1=ev1, ev2=EV(capacity=1.0, max_charge=1.0, max_discharge=1.0, efficiency=1.0, name="ev2"))
    household.current_timestep = 1
    household.history["ev1_at_home"] = {}
    household.history["ev1_at_charging_station"] = {}

    runtime_boundary_rows = []
    for _, row in ev_rows.iterrows():
        timestep = int(row["timestep"])
        status = int(row["status"])

        if status == 0:
            ev1.at_home, ev1.at_charging_station = True, False
        elif status == 1:
            ev1.at_home, ev1.at_charging_station = False, False
        else:
            ev1.at_home, ev1.at_charging_station = False, True

        household.current_timestep = timestep
        household.history["ev1_at_home"][timestep] = 1 if status == 0 else 0
        household.history["ev1_at_charging_station"][timestep] = 1 if status == 2 else 0

        data = _fetch_ev_status_data(household, ev_key)
        phase_ids = _get_phase_id_seq(data["status_history"] + [data["status"]])
        start1, end1, start2, end2 = _get_observed_commute_boundaries(phase_ids)
        runtime_boundary_rows.append(
            {
                "timestep": timestep,
                "status": status,
                "start1": start1,
                "end1": end1,
                "start2": start2,
                "end2": end2,
            }
        )

    runtime_boundary_df = pd.DataFrame(runtime_boundary_rows)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    print("training boundaries")
    print(training_boundary_df)
    print("\nruntime boundaries")
    print(runtime_boundary_df)