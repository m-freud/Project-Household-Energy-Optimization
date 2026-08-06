
# paste this to enable src. imports
from pathlib import Path
import sys
import numpy as np

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.household import Household
from src.simulation.devices.ev import EV
from src.simulation.controllers.mpc.predictors.xgboost.encode_time_cyclic import encode_time_cyclic

from xgboost import XGBClassifier

from src.config import Config


MODEL_FEATURE_COLUMNS = [
    "timestep",
    "status",
    "time_sin",
    "time_cos",
    "steps_in_current_state",
    "phase_id",
    "status_lag_1",
    "status_lag_1_is_pad",
    "status_lag_2",
    "status_lag_2_is_pad",
    "status_lag_4",
    "status_lag_4_is_pad",
    "status_lag_8",
    "status_lag_8_is_pad",
    "start1_earliest",
    "end1_latest",
    "start2_earliest",
    "end2_latest",
    "max_commute_steps_1",
    "max_commute_steps_2",
    "steps_to_start1_earliest",
    "steps_to_end1_latest",
    "steps_to_start2_earliest",
    "steps_to_end2_latest",
    "start1",
    "end1",
    "start2",
    "end2",
    "start1_observed",
    "end1_observed",
    "start2_observed",
    "end2_observed",
    "observed_window_length_1",
    "observed_window_length_2",
    "window_length_slack_1",
    "window_length_slack_2",
]


def _fetch_ev_status_data(household: Household, ev_key: str) -> dict:
    windows = Config.EV_COMMUTE_WINDOWS_ALLOWED[ev_key]
    current_timestep = int(household.current_timestep)

    at_home_history = household.history.get(f"{ev_key}_at_home", {})
    at_station_history = household.history.get(f"{ev_key}_at_charging_station", {})

    # Histories are keyed by timestep in the simulator; keep temporal order for lag features.
    ordered_steps = [
        int(step)
        for step in sorted(set(at_home_history.keys()) | set(at_station_history.keys()))
        if int(step) < current_timestep
    ]
    status_history = [
        int(1 - int(at_home_history.get(step, 0)) + int(at_station_history.get(step, 0)))
        for step in ordered_steps
    ]

    status_now = int(
        1
        - int(getattr(household, f"{ev_key}_at_home", 0))
        + int(getattr(household, f"{ev_key}_at_charging_station", 0))
    )

    status_seq = status_history + [status_now]
    timestep_seq = ordered_steps + [current_timestep]

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

    drive1_steps = [t for t, phase_id in zip(timestep_seq, phase_ids) if phase_id == 1]
    drive2_steps = [t for t, phase_id in zip(timestep_seq, phase_ids) if phase_id == 3]

    start1_time = int(min(drive1_steps)) if drive1_steps else -1
    end1_time = int(max(drive1_steps)) if drive1_steps else -1
    start2_time = int(min(drive2_steps)) if drive2_steps else -1
    end2_time = int(max(drive2_steps)) if drive2_steps else -1

    current_phase_id = int(phase_ids[-1]) if phase_ids else 0

    # Match training semantics: reveal boundaries only after entering later phases.
    start1 = int(start1_time) if current_phase_id > 0 else -1
    end1 = int(end1_time) if current_phase_id > 1 else -1
    start2 = int(start2_time) if current_phase_id > 2 else -1
    end2 = int(end2_time) if current_phase_id > 3 else -1

    return {
        "timestep": int(current_timestep),
        "status": int(status_now),
        "status_history": status_history,
        "start1_earliest": int(windows[0]["earliest_start"]),
        "end1_latest": int(windows[0]["latest_end"]),
        "start2_earliest": int(windows[1]["earliest_start"]),
        "end2_latest": int(windows[1]["latest_end"]),
        "max_commute_steps_1": int(windows[0]["max_unavailable_steps"]),
        "max_commute_steps_2": int(windows[1]["max_unavailable_steps"]),
        "start1": int(start1),
        "end1": int(end1),
        "start2": int(start2),
        "end2": int(end2),
    }



def _build_ev_status_features(ev_status_data) -> dict:
    timestep = int(ev_status_data["timestep"])
    status = int(ev_status_data["status"])
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

    seen_station = any(int(s) == 2 for s in status_seq)
    if status == 2:
        phase = "station"
    elif status == 1 and not seen_station:
        phase = "drive1"
    elif status == 1 and seen_station:
        phase = "drive2"
    elif status == 0 and not seen_station:
        phase = "home1"
    else:
        phase = "home2"

    phase_to_id = {
        "home1": 0,
        "drive1": 1,
        "station": 2,
        "drive2": 3,
        "home2": 4,
    }
    phase_id = int(phase_to_id[phase])

    start1 = int(ev_status_data["start1"])
    end1 = int(ev_status_data["end1"])
    start2 = int(ev_status_data["start2"])
    end2 = int(ev_status_data["end2"])

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
        "timestep": int(timestep),
        "status": int(status),
        "time_sin": float(encode_time_cyclic(timestep, Config.TOTAL_TIMESTEPS_DAY)[0]),
        "time_cos": float(encode_time_cyclic(timestep, Config.TOTAL_TIMESTEPS_DAY)[1]),
        "steps_in_current_state": _steps_in_current_state(),
        "phase_id": int(phase_id),
        "status_lag_1": int(status_lag_1),
        "status_lag_1_is_pad": int(status_lag_1_is_pad),
        "status_lag_2": int(status_lag_2),
        "status_lag_2_is_pad": int(status_lag_2_is_pad),
        "status_lag_4": int(status_lag_4),
        "status_lag_4_is_pad": int(status_lag_4_is_pad),
        "status_lag_8": int(status_lag_8),
        "status_lag_8_is_pad": int(status_lag_8_is_pad),
        "start1_earliest": int(ev_status_data["start1_earliest"]),
        "end1_latest": int(ev_status_data["end1_latest"]),
        "start2_earliest": int(ev_status_data["start2_earliest"]),
        "end2_latest": int(ev_status_data["end2_latest"]),
        "max_commute_steps_1": int(max_commute_steps_1),
        "max_commute_steps_2": int(max_commute_steps_2),
        "steps_to_start1_earliest": _steps_to_boundary(int(ev_status_data["start1_earliest"])),
        "steps_to_end1_latest": _steps_to_boundary(int(ev_status_data["end1_latest"])),
        "steps_to_start2_earliest": _steps_to_boundary(int(ev_status_data["start2_earliest"])),
        "steps_to_end2_latest": _steps_to_boundary(int(ev_status_data["end2_latest"])),
        "start1": int(start1),
        "end1": int(end1),
        "start2": int(start2),
        "end2": int(end2),
        "start1_observed": int(start1_observed),
        "end1_observed": int(end1_observed),
        "start2_observed": int(start2_observed),
        "end2_observed": int(end2_observed),
        "observed_window_length_1": int(observed_window_length_1),
        "observed_window_length_2": int(observed_window_length_2),
        "window_length_slack_1": int(window_length_slack_1),
        "window_length_slack_2": int(window_length_slack_2),
    }
    return features


def _predict_single_ev_status(model: XGBClassifier, household: Household, ev_key: str, horizon: int=96) -> tuple[list[float], list[float]]:
    """
    Predicts the status of a single EV (at_home, at_charging_station) for the given household and horizon.

    Args:
        household (Household): The household for which to predict EV status.
        ev_key (str): The key identifying the EV.
        horizon (int): The number of time steps to predict.

    Returns:
        tuple[list[float], list[float]]: Two lists representing the predicted status of the EV at home and at the charging station.
    """
    if horizon <= 0:
        return [], []

    def _status_to_home_station(status: int) -> tuple[int, int]:
        if status == 0:
            return 1, 0
        if status == 1:
            return 0, 0
        if status == 2:
            return 0, 1
        raise ValueError(f"Unexpected EV status class: {status}")

    at_home_pred: list[float] = []
    at_station_pred: list[float] = []

    history_home_key = f"{ev_key}_at_home"
    history_station_key = f"{ev_key}_at_charging_station"
    ev_obj = getattr(household, ev_key)

    original_timestep = int(household.current_timestep)
    original_at_home = bool(getattr(ev_obj, "at_home", False))
    original_at_station = bool(getattr(ev_obj, "at_charging_station", False))
    original_home_history = dict(household.history.get(history_home_key, {}))
    original_station_history = dict(household.history.get(history_station_key, {}))

    sim_home_history = dict(original_home_history)
    sim_station_history = dict(original_station_history)

    current_status = int(1 - int(original_at_home) + int(original_at_station))
    current_timestep = original_timestep

    current_home, current_station = _status_to_home_station(current_status)
    at_home_pred.append(float(current_home))
    at_station_pred.append(float(current_station))

    try:
        for _ in range(horizon - 1):
            household.current_timestep = current_timestep
            household.history[history_home_key] = sim_home_history
            household.history[history_station_key] = sim_station_history
            ev_obj.at_home = bool(current_home)
            ev_obj.at_charging_station = bool(current_station)

            ev_status_data = _fetch_ev_status_data(household, ev_key)
            features = _build_ev_status_features(ev_status_data)

            model_input = np.asarray([[features[column] for column in MODEL_FEATURE_COLUMNS]], dtype=float)

            predicted_status = int(model.predict(model_input)[0])

            # Move the current status into history before advancing time.
            sim_home_history[current_timestep] = int(current_home)
            sim_station_history[current_timestep] = int(current_station)

            current_status = predicted_status
            current_timestep += 1
            current_home, current_station = _status_to_home_station(current_status)

            at_home_pred.append(float(current_home))
            at_station_pred.append(float(current_station))
    finally:
        household.current_timestep = original_timestep
        household.history[history_home_key] = original_home_history
        household.history[history_station_key] = original_station_history
        ev_obj.at_home = bool(original_at_home)
        ev_obj.at_charging_station = bool(original_at_station)

    return at_home_pred, at_station_pred
    

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


if __name__ == "__main__":
    from pprint import pprint
    from training.xgboost.features.ev_status_features import get_ev_status_features

    household_id = 1
    timestep = 42
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

    ev1 = EV(capacity=1.0, max_charge=1.0, max_discharge=1.0, efficiency=1.0, name="ev1")
    ev2 = EV(capacity=1.0, max_charge=1.0, max_discharge=1.0, efficiency=1.0, name="ev2")
    household = Household(player_id=household_id, start_time=timestep, ev1=ev1, ev2=ev2)
    household.current_timestep = timestep

    status_history_rows = ev_rows[ev_rows["timestep"] < timestep]
    for _, row in status_history_rows.iterrows():
        t = int(row["timestep"])
        status = int(row["status"])
        if status == 0:
            at_home, at_station = 1, 0
        elif status == 1:
            at_home, at_station = 0, 0
        else:
            at_home, at_station = 0, 1
        household.history["ev1_at_home"][t] = at_home
        household.history["ev1_at_charging_station"][t] = at_station

    status_now = int(ev_rows.loc[ev_rows["timestep"] == timestep, "status"].iloc[0])
    if status_now == 0:
        ev1.at_home, ev1.at_charging_station = True, False
    elif status_now == 1:
        ev1.at_home, ev1.at_charging_station = False, False
    else:
        ev1.at_home, ev1.at_charging_station = False, True

    data = _fetch_ev_status_data(household, ev_key)
    features = _build_ev_status_features(data)

    print(f"EV status test snapshot: household={household_id}, ev={ev_key}, timestep={timestep}")
    print("\n_fetch_ev_status_data output:")
    pprint(data, sort_dicts=False)
    print("\n_build_ev_status_features output:")
    pprint(features, sort_dicts=False)
