
from src.runtime_conig import RuntimeConfig
from src.simulation.controllers.mpc.predictors.ml.helpers.encode_time_cyclic import encode_time_cyclic
from src.simulation.controllers.mpc.predictors.ml.model_interface import TRegressor
from src.simulation.controllers.mpc.predictors.shared import make_band
from src.simulation.household import Household
import numpy as np
from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig



def _build_pv_gen_features(
    current_timestep: int,
    current_pv_gen: float,
    pv_history: list[float],
    daylight_start: int,
    daylight_end: int,
    round_values: bool = False,
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

    pv_delta_1 = current_pv_gen - lag_1 if lag_1_pad == 0 else 0.0
    pv_delta_2 = lag_1 - lag_2 if (lag_1_pad == 0 and lag_2_pad == 0) else 0.0
    pv_accel = pv_delta_1 - pv_delta_2

    steps_to_daylight_start = daylight_start - current_timestep + 1 if current_timestep <= daylight_start else daylight_start - current_timestep
    steps_to_daylight_end = daylight_end - current_timestep + 1 if current_timestep <= daylight_end else daylight_end - current_timestep

    time_sin, time_cos = encode_time_cyclic(current_timestep)

    features = {
        "timestep": current_timestep,
        "pv_gen": current_pv_gen,
        "time_sin": time_sin,
        "time_cos": time_cos,
        "pv_lag_1": lag_1,
        "pv_lag_1_is_pad": lag_1_pad,
        "pv_lag_2": lag_2,
        "pv_lag_2_is_pad": lag_2_pad,
        "pv_lag_4": lag_4,
        "pv_lag_4_is_pad": lag_4_pad,
        "pv_lag_8": lag_8,
        "pv_lag_8_is_pad": lag_8_pad,
        "pv_lag_12": lag_12,
        "pv_lag_12_is_pad": lag_12_pad,
        "pv_ma_2": _rolling_mean(2),
        "pv_ma_4": _rolling_mean(4),
        "pv_ma_8": _rolling_mean(8),
        "pv_ma_16": _rolling_mean(16),
        "pv_std_4": _rolling_std(4),
        "pv_std_8": _rolling_std(8),
        "pv_delta_1": pv_delta_1,
        "pv_delta_2": pv_delta_2,
        "pv_accel": pv_accel,
        "steps_to_daylight_start": steps_to_daylight_start,
        "steps_to_daylight_end": steps_to_daylight_end,
    }

    if round_values:
        for key, value in list(features.items()):
            if isinstance(value, float):
                features[key] = round(value, 3)

    return features


def _try_bypass(current_time: int) -> float | None:
    # post-midnight bypass is implied
    if current_time < RuntimeConfig.PV_GENERATION_WINDOW_ALLOWED["earliest_start"] \
       or current_time > RuntimeConfig.PV_GENERATION_WINDOW_ALLOWED["latest_end"]:
        return 0.0
    return None


def _predict_pv_gen(
        model: TRegressor,
        household: Household,
        horizon: int = 96
        ) -> list[float]:
    
    if not household.has_pv:
        return [0.0] * horizon

    daylight_start = RuntimeConfig.PV_GENERATION_WINDOW_ALLOWED["earliest_start"]
    daylight_end = RuntimeConfig.PV_GENERATION_WINDOW_ALLOWED["latest_end"]

    # current values
    current_timestep = household.current_timestep
    current_pv_gen = household.pv_gen

    # init sim history
    sim_pv_history = list(household.history["pv_gen"].values())

    # init prediciton
    pv_pred: list[float] = [current_pv_gen]

    for _ in range(horizon - 1):
        bypass = _try_bypass(current_timestep)
        if bypass is not None:
            current_pv_gen = float(bypass)
            pv_pred.append(current_pv_gen)
            current_timestep += 1
            continue


        all_features = _build_pv_gen_features(
            current_timestep=current_timestep,
            current_pv_gen=current_pv_gen,
            pv_history=sim_pv_history,
            daylight_start=daylight_start,
            daylight_end=daylight_end,
        )

        model_family_name = ModelConfig.get_model_family_name(model)
        model_features = ModelConfig.MODEL_FEATURES_BY_FAMILY[model_family_name]["pv_gen"]

        # ensure completness of features for the model
        for f in model_features:
            if f not in all_features:
                raise ValueError(f"Missing required feature: {f}")

        # ensure correct order of features (as in model_features)
        model_input = [all_features[f] for f in model_features]

        # update sim hist before next prediction
        sim_pv_history.append(float(current_pv_gen))

        current_pv_gen = model.predict([model_input])[0] # type: ignore
        pv_pred.append(current_pv_gen)

        # incr time
        current_timestep += 1

    return pv_pred


def predict_pv_gen(model: TRegressor, household: Household, horizon: int, interval_width_frct: float = 0.0) -> dict[str, list[float]]:
    pv_gen = _predict_pv_gen(model=model, household=household, horizon=horizon)
    pv_gen_lb, pv_gen_ub = make_band(pv_gen, interval_width_frct)

    return {
        "pv_gen": pv_gen,
        "pv_gen_lb": pv_gen_lb,
        "pv_gen_ub": pv_gen_ub,
    }