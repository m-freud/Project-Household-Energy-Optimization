from pathlib import Path
import sys

import numpy as np

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.simulation.household import Household
from src.simulation.controllers.mpc.predictors.shared.make_band import make_band
from src.simulation.controllers.mpc.predictors.xgboost.encode_time_cyclic import encode_time_cyclic
from xgboost import XGBRegressor


def _count_evs_at_home(
    predicted_ev_status: dict[str, list[int]],
    prediction_index: int,
) -> int:
    ev1_home_seq = predicted_ev_status.get("ev1_at_home", [])
    ev2_home_seq = predicted_ev_status.get("ev2_at_home", [])

    ev1_at_home = ev1_home_seq[prediction_index]
    ev2_at_home = ev2_home_seq[prediction_index]
    
    return ev1_at_home + ev2_at_home


def _build_base_load_features(
    current_timestep: int,
    current_base_load: float,
    base_load_history: list[float], # dict is converted to list for calculations
    n_evs_at_home: int,
    round_values: bool = False,
) -> dict:
    base_load_seq = base_load_history + [current_base_load]

    def _lag(lag: int) -> tuple[float, int]:
        idx = len(base_load_seq) - 1 - lag
        if idx >= 0:
            return float(base_load_seq[idx]), 0
        return -1.0, 1

    def _rolling_mean(window: int) -> float:
        return float(np.mean(np.asarray(base_load_seq[-window:], dtype=float)))

    def _rolling_std(window: int) -> float:
        return float(np.std(np.asarray(base_load_seq[-window:], dtype=float), ddof=0))

    lag_1, lag_1_pad = _lag(1)
    lag_2, lag_2_pad = _lag(2)
    lag_4, lag_4_pad = _lag(4)
    lag_8, lag_8_pad = _lag(8)
    lag_12, lag_12_pad = _lag(12)

    base_load_delta_1 = current_base_load - lag_1 if lag_1_pad == 0 else 0.0
    base_load_delta_2 = lag_1 - lag_2 if (lag_1_pad == 0 and lag_2_pad == 0) else 0.0
    base_load_accel = base_load_delta_1 - base_load_delta_2

    time_sin, time_cos = encode_time_cyclic(current_timestep)

    features = {
        "timestep": current_timestep,
        "base_load": current_base_load,
        "n_evs_at_home": n_evs_at_home,
        "time_sin": time_sin,
        "time_cos": time_cos,
        "base_load_lag_1": lag_1,
        "base_load_lag_1_is_pad": lag_1_pad,
        "base_load_lag_2": lag_2,
        "base_load_lag_2_is_pad": lag_2_pad,
        "base_load_lag_4": lag_4,
        "base_load_lag_4_is_pad": lag_4_pad,
        "base_load_lag_8": lag_8,
        "base_load_lag_8_is_pad": lag_8_pad,
        "base_load_lag_12": lag_12,
        "base_load_lag_12_is_pad": lag_12_pad,
        "base_load_ma_2": _rolling_mean(2),
        "base_load_ma_4": _rolling_mean(4),
        "base_load_ma_8": _rolling_mean(8),
        "base_load_ma_16": _rolling_mean(16),
        "base_load_std_4": _rolling_std(4),
        "base_load_std_8": _rolling_std(8),
        "base_load_delta_1": base_load_delta_1,
        "base_load_delta_2": base_load_delta_2,
        "base_load_accel": base_load_accel,
    }

    if round_values:
        for key, value in list(features.items()):
            if isinstance(value, float):
                features[key] = round(value, 3)

    return features


def _try_bypass(current_timestep: int) -> float | None:
    if current_timestep >= 96:
        return 0.0
    return None


def _predict_base_load(
    model: XGBRegressor,
    household: Household,
    horizon: int,
    predicted_ev_status: dict[str, list[int]],
) -> list[float]:
    # current values
    current_timestep = household.current_timestep
    current_base_load = household.base_load

    # init sim history
    # we use a list here for convenience
    sim_base_load_history = list(household.history["base_load"].values())

    # init pred
    base_load_pred: list[float] = [current_base_load]

    for prediction_index in range(horizon - 1):
        bypass = _try_bypass(current_timestep)
        if bypass is not None:
            current_base_load = bypass
            base_load_pred.append(current_base_load)
            current_timestep += 1
            continue

        n_evs_at_home = _count_evs_at_home( # predictions are 0-indexed so we can just use this range
            predicted_ev_status=predicted_ev_status,
            prediction_index=prediction_index,
        )

        features = _build_base_load_features(
            current_timestep=current_timestep,
            current_base_load=current_base_load,
            base_load_history=sim_base_load_history,
            n_evs_at_home=n_evs_at_home,
        )

        # ensure completness of features for the model
        for f in Config.XGB_FEATURES["BASE_LOAD"]:
            if f not in features:
                raise ValueError(f"Missing required feature: {f}")

        # ensure correct order of features
        model_input = [features[f] for f in Config.XGB_FEATURES["BASE_LOAD"]]

        # update sim hist before next prediction
        sim_base_load_history.append(current_base_load)

        # get prediction and append
        current_base_load = model.predict([model_input])[0]
        base_load_pred.append(current_base_load)

        # incr time
        current_timestep += 1

    return base_load_pred


def predict_base_load(
    model: XGBRegressor,
    household: Household,
    horizon: int,
    predicted_ev_status: dict[str, list[int]],
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