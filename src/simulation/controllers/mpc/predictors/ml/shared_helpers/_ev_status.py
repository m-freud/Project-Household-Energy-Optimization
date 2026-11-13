from collections.abc import Collection

from src.runtime_config import RuntimeConfig
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._misc import encode_time_cyclic


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


def _build_ev_status_features(
    ev_status_data,
    required_features: Collection[str] | None = None,
) -> dict:
    """
    build features ready to be used in classification model
    """
    requested = set(required_features) if required_features is not None else None

    def needs(name: str) -> bool:
        return requested is None or name in requested

    timestep = ev_status_data["timestep"]
    status = ev_status_data["status"]
    status_history = list(ev_status_data.get("status_history", []))
    status_seq = status_history + [status]

    def _lag(lag: int) -> tuple[int, int]:
        idx = len(status_seq) - 1 - lag
        if idx >= 0:
            return int(status_seq[idx]), 0
        return -1, 1

    max_commute_steps_1 = int(ev_status_data["max_commute_steps_1"])
    max_commute_steps_2 = int(ev_status_data["max_commute_steps_2"])

    def _steps_to_boundary(boundary: int) -> int:
        if timestep <= boundary:
            return int(boundary - timestep + 1)
        return int(boundary - timestep)

    boundary_features = {
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
    }
    needs_boundaries = requested is None or bool(requested & boundary_features)
    if needs_boundaries:
        phase_ids = _get_phase_id_seq(status_seq)
        start1, end1, start2, end2 = _get_observed_commute_boundaries(phase_ids)

    features: dict[str, int | float] = {}
    if needs("timestep"):
        features["timestep"] = timestep
    if needs("status"):
        features["status"] = status
    if needs("time_sin") or needs("time_cos"):
        time_sin, time_cos = encode_time_cyclic(timestep)
        if needs("time_sin"):
            features["time_sin"] = float(time_sin)
        if needs("time_cos"):
            features["time_cos"] = float(time_cos)

    if needs("steps_in_current_state"):
        steps = 1
        for previous in reversed(status_seq[:-1]):
            if int(previous) != status:
                break
            steps += 1
        features["steps_in_current_state"] = steps

    if needs("phase_id"):
        seen_station = 2 in status_seq
        if status == 2:
            phase_id = 2
        elif status == 1:
            phase_id = 3 if seen_station else 1
        else:
            phase_id = 4 if seen_station else 0
        features["phase_id"] = phase_id

    for lag in (1, 2, 4, 8):
        lag_name = f"status_lag_{lag}"
        pad_name = f"{lag_name}_is_pad"
        if needs(lag_name) or needs(pad_name):
            lag_value, lag_is_pad = _lag(lag)
            if needs(lag_name):
                features[lag_name] = lag_value
            if needs(pad_name):
                features[pad_name] = lag_is_pad

    config_features = {
        "start1_earliest": ev_status_data["start1_earliest"],
        "end1_latest": ev_status_data["end1_latest"],
        "start2_earliest": ev_status_data["start2_earliest"],
        "end2_latest": ev_status_data["end2_latest"],
        "max_commute_steps_1": max_commute_steps_1,
        "max_commute_steps_2": max_commute_steps_2,
    }
    for name, value in config_features.items():
        if needs(name):
            features[name] = value

    for name, config_name in (
        ("steps_to_start1_earliest", "start1_earliest"),
        ("steps_to_end1_latest", "end1_latest"),
        ("steps_to_start2_earliest", "start2_earliest"),
        ("steps_to_end2_latest", "end2_latest"),
    ):
        if needs(name):
            features[name] = _steps_to_boundary(ev_status_data[config_name])

    if needs_boundaries:
        observed_values = {
            "start1": start1,
            "end1": end1,
            "start2": start2,
            "end2": end2,
            "start1_observed": int(start1 != -1),
            "end1_observed": int(end1 != -1),
            "start2_observed": int(start2 != -1),
            "end2_observed": int(end2 != -1),
            "observed_window_length_1": (
                int(end1 - start1 + 1) if end1 != -1 else -1
            ),
            "observed_window_length_2": (
                int(end2 - start2 + 1) if end2 != -1 else -1
            ),
        }
        observed_values["window_length_slack_1"] = (
            int(max_commute_steps_1 - observed_values["observed_window_length_1"])
            if observed_values["observed_window_length_1"] != -1
            else -1
        )
        observed_values["window_length_slack_2"] = (
            int(max_commute_steps_2 - observed_values["observed_window_length_2"])
            if observed_values["observed_window_length_2"] != -1
            else -1
        )
        for name, value in observed_values.items():
            if needs(name):
                features[name] = value

    return features


def _try_bypass( #TODO fix according to ev1/ev2 setup
    current_time: int,
    ev_key: str,
    status_history: list[int],
    current_status: int,
) -> int | None:
    """Return a deterministic next status using the configured EV commute rules."""
    windows = RuntimeConfig.EV_COMMUTE_WINDOWS_ALLOWED[ev_key]
    first_window, second_window = windows
    target_time = current_time + 1
    if target_time > 96:
        return 0

    phase_id = _get_phase_id_seq([*status_history, current_status])[-1]
    steps_in_current_state = 1
    for previous_status in reversed(status_history):
        if previous_status != current_status:
            break
        steps_in_current_state += 1

    if (
        phase_id == 1
        and steps_in_current_state >= first_window["max_unavailable_steps"]
    ):
        return 2
    if (
        phase_id == 3
        and steps_in_current_state >= second_window["max_unavailable_steps"]
    ):
        return 0

    if phase_id == 4 or target_time > second_window["latest_end"]:
        return 0
    if target_time < first_window["earliest_start"]:
        return 0

    if (
        target_time > first_window["latest_end"]
        and target_time < second_window["earliest_start"]
    ):
        if phase_id in (1, 2):
            return 2
        if phase_id == 0:
            return 0

    return None
