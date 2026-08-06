from pathlib import Path
import sys

import numpy as np
import pandas as pd

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.simulation.household import Household
from src.simulation.controllers.mpc.predictors.shared.make_band import make_band
from src.simulation.controllers.mpc.predictors.xgboost.encode_time_cyclic import encode_time_cyclic
from xgboost import XGBRegressor


MODEL_FEATURE_COLUMNS = [
    "timestep",
    "base_load",
    "n_evs_at_home",
    "time_sin",
    "time_cos",
    "base_load_lag_1",
    "base_load_lag_1_is_pad",
    "base_load_lag_2",
    "base_load_lag_2_is_pad",
    "base_load_lag_4",
    "base_load_lag_4_is_pad",
    "base_load_lag_8",
    "base_load_lag_8_is_pad",
    "base_load_lag_12",
    "base_load_lag_12_is_pad",
    "base_load_ma_2",
    "base_load_ma_4",
    "base_load_ma_8",
    "base_load_ma_16",
    "base_load_std_4",
    "base_load_std_8",
    "base_load_delta_1",
    "base_load_delta_2",
    "base_load_accel",
]


def _history_values(household: Household, key: str, current_timestep: int) -> list[float]:
    history = household.history.get(key, {})
    if not history:
        return []

    ordered_steps = sorted(int(step) for step in history.keys() if int(step) < int(current_timestep))
    return [float(history[step]) for step in ordered_steps]


def _count_evs_at_home(
    household: Household,
    predicted_ev_status: dict[str, list[float]] | None = None,
    prediction_index: int | None = None,
) -> int:
    if predicted_ev_status is not None and prediction_index is not None:
        ev1_home_seq = predicted_ev_status.get("ev1_at_home", [])
        ev2_home_seq = predicted_ev_status.get("ev2_at_home", [])

        ev1_at_home = int(float(ev1_home_seq[prediction_index])) if prediction_index < len(ev1_home_seq) else 0
        ev2_at_home = int(float(ev2_home_seq[prediction_index])) if prediction_index < len(ev2_home_seq) else 0
        return int(ev1_at_home + ev2_at_home)

    ev1_at_home = int(bool(getattr(household.ev1, "at_home", False))) if getattr(household, "ev1", None) else 0
    ev2_at_home = int(bool(getattr(household.ev2, "at_home", False))) if getattr(household, "ev2", None) else 0
    return int(ev1_at_home + ev2_at_home)


def _build_base_load_features(
    household: Household,
    current_timestep: int,
    current_base_load: float,
    base_load_history: list[float],
    n_evs_at_home: int,
) -> dict:
    pv_seq = list(base_load_history) + [float(current_base_load)]

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

    base_load_delta_1 = float(current_base_load - lag_1) if lag_1_pad == 0 else 0.0
    base_load_delta_2 = float(lag_1 - lag_2) if (lag_1_pad == 0 and lag_2_pad == 0) else 0.0
    base_load_accel = float(base_load_delta_1 - base_load_delta_2)

    time_sin, time_cos = encode_time_cyclic(current_timestep, Config.TOTAL_TIMESTEPS_DAY)

    features = {
        "timestep": int(current_timestep),
        "base_load": float(current_base_load),
        "n_evs_at_home": int(n_evs_at_home),
        "time_sin": float(time_sin),
        "time_cos": float(time_cos),
        "base_load_lag_1": float(lag_1),
        "base_load_lag_1_is_pad": int(lag_1_pad),
        "base_load_lag_2": float(lag_2),
        "base_load_lag_2_is_pad": int(lag_2_pad),
        "base_load_lag_4": float(lag_4),
        "base_load_lag_4_is_pad": int(lag_4_pad),
        "base_load_lag_8": float(lag_8),
        "base_load_lag_8_is_pad": int(lag_8_pad),
        "base_load_lag_12": float(lag_12),
        "base_load_lag_12_is_pad": int(lag_12_pad),
        "base_load_ma_2": _rolling_mean(2),
        "base_load_ma_4": _rolling_mean(4),
        "base_load_ma_8": _rolling_mean(8),
        "base_load_ma_16": _rolling_mean(16),
        "base_load_std_4": _rolling_std(4),
        "base_load_std_8": _rolling_std(8),
        "base_load_delta_1": float(base_load_delta_1),
        "base_load_delta_2": float(base_load_delta_2),
        "base_load_accel": float(base_load_accel),
    }

    for key, value in list(features.items()):
        if isinstance(value, float):
            features[key] = round(value, 3)

    return features


def _predict_base_load(
    model: XGBRegressor,
    household: Household,
    horizon: int = 96,
    predicted_ev_status: dict[str, list[float]] | None = None,
) -> list[float]:
    if horizon <= 0:
        return []

    original_timestep = int(household.current_timestep)
    current_timestep = int(original_timestep)

    current_base_load = float(household.base_load)
    base_load_pred: list[float] = [current_base_load]

    base_load_history = _history_values(household, "base_load", current_timestep)

    for prediction_index in range(horizon - 1):
        n_evs_at_home = _count_evs_at_home(
            household,
            predicted_ev_status=predicted_ev_status,
            prediction_index=prediction_index,
        )

        features = _build_base_load_features(
            household=household,
            current_timestep=current_timestep,
            current_base_load=current_base_load,
            base_load_history=base_load_history,
            n_evs_at_home=n_evs_at_home,
        )

        feature_row = pd.DataFrame([features])
        model_input = feature_row[MODEL_FEATURE_COLUMNS]

        next_base_load = float(model.predict(model_input)[0])
        next_base_load = max(0.0, next_base_load)

        base_load_history.append(float(current_base_load))
        current_timestep += 1
        current_base_load = float(next_base_load)
        base_load_pred.append(float(current_base_load))

    return base_load_pred


def predict_base_load(
    model: XGBRegressor,
    household: Household,
    horizon: int,
    predicted_ev_status: dict[str, list[float]] | None = None,
    interval_width_frct: float = 0.0,
) -> dict[str, list[float]]:
    base_load = _predict_base_load(
        model=model,
        household=household,
        horizon=horizon,
        predicted_ev_status=predicted_ev_status,
    )
    base_load_lb, base_load_ub = make_band(base_load, interval_width_frct)

    return {
        "base_load": base_load,
        "base_load_lb": base_load_lb,
        "base_load_ub": base_load_ub,
    }