import numpy as np
from src.runtime_config import RuntimeConfig
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._misc import encode_time_cyclic

LAG_STEPS = (1, 2, 4, 8, 12, 16, 24, 32, 48)
MA_WINDOWS = (2, 4, 8, 16, 24, 32, 48)
STD_WINDOWS = (4, 8, 16, 32, 48)
DELTA_STEPS = (1, 2, 4, 8, 16)
ACC_STEPS = (2, 4, 8, 16)


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

    lags: dict[int, float] = {}
    lag_pads: dict[int, int] = {}
    for step in LAG_STEPS:
        lags[step], lag_pads[step] = _lag(step)

    deltas = {
        step: (current_pv_gen - lags[step] if lag_pads[step] == 0 else 0.0)
        for step in DELTA_STEPS
    }
    accs = {step: deltas[step] - deltas[1] for step in ACC_STEPS}

    steps_to_daylight_start = daylight_start - current_timestep + 1 if current_timestep <= daylight_start else daylight_start - current_timestep
    steps_to_daylight_end = daylight_end - current_timestep + 1 if current_timestep <= daylight_end else daylight_end - current_timestep

    time_sin, time_cos = encode_time_cyclic(current_timestep)

    features = {
        "timestep": current_timestep,
        "pv_gen": current_pv_gen,
        "time_sin": time_sin,
        "time_cos": time_cos,
    }

    for step in LAG_STEPS:
        features[f"pv_lag_{step}"] = lags[step]
        features[f"pv_lag_{step}_is_pad"] = lag_pads[step]

    for window in MA_WINDOWS:
        features[f"pv_ma_{window}"] = _rolling_mean(window)

    for window in STD_WINDOWS:
        features[f"pv_std_{window}"] = _rolling_std(window)

    for step in DELTA_STEPS:
        features[f"pv_delta_{step}"] = deltas[step]

    for step in ACC_STEPS:
        features[f"pv_acc_{step}"] = accs[step]

    features["steps_to_daylight_start"] = steps_to_daylight_start
    features["steps_to_daylight_end"] = steps_to_daylight_end

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
