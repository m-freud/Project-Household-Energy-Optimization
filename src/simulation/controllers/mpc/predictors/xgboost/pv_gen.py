from pathlib import Path
import sys
import pandas as pd
import numpy as np

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.household import Household
from xgboost import XGBRegressor
from src.config import Config
from src.simulation.controllers.mpc.predictors.xgboost.encode_time_cyclic import encode_time_cyclic
from src.simulation.controllers.mpc.predictors.shared.make_band import make_band


def _fetch_pv_gen_profiles(
    household: Household,
    current_timestep: int,
    current_pv_gen: float,
    pv_history: dict[int, float],
) -> dict:
    pv_window = getattr(Config, "PV_GENERATION_WINDOW_ALLOWED", None)
    if pv_window is None:
        pv_window = getattr(Config, "PV_GENERATION_WINDOW_OBSERVED", None)
    if pv_window is None:
        raise ValueError("Config must define PV_GENERATION_WINDOW_ALLOWED or PV_GENERATION_WINDOW_OBSERVED")

    ordered_steps = sorted(int(step) for step in pv_history.keys() if int(step) < int(current_timestep))
    pv_series_history = [float(pv_history[step]) for step in ordered_steps]

    return {
        "household_id": int(household.player_id),
        "timestep": int(current_timestep),
        "pv_gen": float(current_pv_gen),
        "pv_history": pv_series_history,
        "daylight_start": int(pv_window["earliest_start"]),
        "daylight_end": int(pv_window["latest_end"]),
    }


def _build_pv_gen_features(
    current_timestep: int,
    current_pv_gen: float,
    pv_history: list[float],
    daylight_start: int,
    daylight_end: int,
) -> dict:
    pv_seq = pv_history + [current_pv_gen]

    def _lag(lag: int) -> tuple[float, int]:
        idx = len(pv_seq) - 1 - lag
        if idx >= 0:
            return float(pv_seq[idx]), 0
        return -1.0, 1

    def _rolling_mean(window: int) -> float:
        return float(np.mean(np.asarray(pv_seq[-window:], dtype=float)))

    def _rolling_std(window: int) -> float:
        return float(np.std(np.asarray(pv_seq[-window:], dtype=float), ddof=0))

    lag_1, lag_1_pad = _lag(1)
    lag_2, lag_2_pad = _lag(2)
    lag_4, lag_4_pad = _lag(4)
    lag_8, lag_8_pad = _lag(8)
    lag_12, lag_12_pad = _lag(12)

    pv_delta_1 = float(current_pv_gen - lag_1) if lag_1_pad == 0 else 0.0
    pv_delta_2 = float(lag_1 - lag_2) if (lag_1_pad == 0 and lag_2_pad == 0) else 0.0
    pv_accel = float(pv_delta_1 - pv_delta_2)

    steps_to_daylight_start = int(daylight_start - current_timestep + 1) if current_timestep <= daylight_start else int(daylight_start - current_timestep)
    steps_to_daylight_end = int(daylight_end - current_timestep + 1) if current_timestep <= daylight_end else int(daylight_end - current_timestep)

    time_sin, time_cos = encode_time_cyclic(current_timestep)

    features = {
        "timestep": int(current_timestep),
        "pv_gen": float(current_pv_gen),
        "time_sin": float(time_sin),
        "time_cos": float(time_cos),
        "pv_lag_1": float(lag_1),
        "pv_lag_1_is_pad": int(lag_1_pad),
        "pv_lag_2": float(lag_2),
        "pv_lag_2_is_pad": int(lag_2_pad),
        "pv_lag_4": float(lag_4),
        "pv_lag_4_is_pad": int(lag_4_pad),
        "pv_lag_8": float(lag_8),
        "pv_lag_8_is_pad": int(lag_8_pad),
        "pv_lag_12": float(lag_12),
        "pv_lag_12_is_pad": int(lag_12_pad),
        "pv_ma_2": _rolling_mean(2),
        "pv_ma_4": _rolling_mean(4),
        "pv_ma_8": _rolling_mean(8),
        "pv_ma_16": _rolling_mean(16),
        "pv_std_4": _rolling_std(4),
        "pv_std_8": _rolling_std(8),
        "pv_delta_1": float(pv_delta_1),
        "pv_delta_2": float(pv_delta_2),
        "pv_accel": float(pv_accel),
        "steps_to_daylight_start": int(steps_to_daylight_start),
        "steps_to_daylight_end": int(steps_to_daylight_end),
    }

    # Keep inference numerics aligned with rounded training features.
    for key, value in list(features.items()):
        if isinstance(value, float):
            features[key] = round(value, 3)

    return features


def _predict_pv_gen(model: XGBRegressor, household: Household, horizon: int = 96) -> list[float]:
    daylight_start = Config.PV_GENERATION_WINDOW_ALLOWED["earliest_start"]
    daylight_end = Config.PV_GENERATION_WINDOW_ALLOWED["latest_end"]

    # current values
    current_timestep = household.current_timestep
    current_pv_gen = household.pv_gen

    # init sim history
    sim_pv_history = list(household.history["pv_gen"].values())

    # init prediciton
    pv_pred: list[float] = [current_pv_gen]

    for _ in range(horizon - 1):
        features = _build_pv_gen_features(
            current_timestep=current_timestep,
            current_pv_gen=current_pv_gen,
            pv_history=sim_pv_history,
            daylight_start=daylight_start,
            daylight_end=daylight_end,
        )

        # ensure completness of features for the model
        for f in Config.XGB_FEATURES["PV_GEN"]:
                if f not in features:
                    raise ValueError(f"Missing feature '{f}' in features dictionary.")

        # ensure correct order of features
        model_input = [features[f] for f in Config.XGB_FEATURES["PV_GEN"]]

        # update sim hist before next prediction
        sim_pv_history.append(float(current_pv_gen))

        # get prediction and append
        current_pv_gen = float(model.predict([model_input])[0])
        pv_pred.append(float(current_pv_gen))

        # incr time
        current_timestep += 1

    return pv_pred


def predict_pv_gen(model: XGBRegressor, household: Household, horizon: int, interval_width_frct: float = 0.0) -> dict[str, list[float]]:
    pv_gen = _predict_pv_gen(model=model, household=household, horizon=horizon)
    pv_gen_lb, pv_gen_ub = make_band(pv_gen, interval_width_frct)

    return {
        "pv_gen": pv_gen,
        "pv_gen_lb": pv_gen_lb,
        "pv_gen_ub": pv_gen_ub,
    }