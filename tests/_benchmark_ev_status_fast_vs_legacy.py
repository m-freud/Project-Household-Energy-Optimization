import time
from pathlib import Path
import sys

import numpy as np


repo_root = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents] if (p / "src").exists()), Path.cwd().resolve())
sys.path.insert(0, str(repo_root))

from simulation.controllers.mpc.predictors.xgboost_legacy.helpers.ev_status import (
    _build_ev_status_features,
    _fetch_ev_status_data,
    _predict_single_ev_status,
)
from src.simulation.devices.ev import EV
from src.simulation.household import Household


class DeterministicModel:
    def __init__(self):
        self._state = 0

    def predict(self, X):
        _ = X
        result = self._state % 3
        self._state += 1
        return np.asarray([result], dtype=int)


def _legacy_predict_single_ev_status(model, household: Household, ev_key: str, horizon: int = 96):
    if horizon <= 0:
        return [], []

    def _status_to_home_station(status: int):
        if status == 0:
            return 1, 0
        if status == 1:
            return 0, 0
        if status == 2:
            return 0, 1
        raise ValueError(f"Unexpected EV status class: {status}")

    at_home_pred = []
    at_station_pred = []

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
            model_input = np.asarray([[features[column] for column in features]], dtype=float)

            predicted_status = int(model.predict(model_input)[0])

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


def _build_household() -> Household:
    ev1 = EV(capacity=20.0, max_charge=7.0, max_discharge=7.0, efficiency=0.95, name="ev1")
    ev2 = EV(capacity=20.0, max_charge=7.0, max_discharge=7.0, efficiency=0.95, name="ev2")
    household = Household(player_id=1, start_time=120, ev1=ev1, ev2=ev2)
    household.current_timestep = 120

    status_cycle = [0, 1, 2, 1, 0]
    for t in range(1, 120):
        status = status_cycle[t % len(status_cycle)]
        if status == 0:
            home, station = 1, 0
        elif status == 1:
            home, station = 0, 0
        else:
            home, station = 0, 1
        household.history["ev1_at_home"][t] = home
        household.history["ev1_at_charging_station"][t] = station

    last_status = status_cycle[(120 - 1) % len(status_cycle)]
    if last_status == 0:
        household.ev1.at_home = True
        household.ev1.at_charging_station = False
    elif last_status == 1:
        household.ev1.at_home = False
        household.ev1.at_charging_station = False
    else:
        household.ev1.at_home = False
        household.ev1.at_charging_station = True

    return household


def main():
    horizon = 96
    repeats = 300

    household_a = _build_household()
    household_b = _build_household()

    out_legacy = _legacy_predict_single_ev_status(DeterministicModel(), household_a, "ev1", horizon)
    out_fast = _predict_single_ev_status(DeterministicModel(), household_b, "ev1", horizon)
    if out_legacy != out_fast:
        raise AssertionError("Legacy and fast outputs differ")

    t0 = time.perf_counter()
    for _ in range(repeats):
        _legacy_predict_single_ev_status(DeterministicModel(), _build_household(), "ev1", horizon)
    legacy_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    for _ in range(repeats):
        _predict_single_ev_status(DeterministicModel(), _build_household(), "ev1", horizon)
    fast_elapsed = time.perf_counter() - t1

    print(f"legacy_total_s {legacy_elapsed:.6f}")
    print(f"fast_total_s {fast_elapsed:.6f}")
    print(f"speedup_x {(legacy_elapsed / fast_elapsed):.3f}")


if __name__ == "__main__":
    main()
