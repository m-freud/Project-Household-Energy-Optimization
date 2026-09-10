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
