import numpy as np
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._misc import encode_time_cyclic

# how far back we look; deeper lags/ma/std are mostly pad early in a run since
# household.history starts empty at start_time (see model_config.py notes)
LAG_STEPS = (1, 2, 4, 8, 12, 16, 24, 32, 48)
MA_WINDOWS = (2, 4, 8, 16, 24, 32, 48)
STD_WINDOWS = (4, 8, 16, 32, 48)
DELTA_STEPS = (1, 2, 4, 8, 16)
ACC_STEPS = (2, 4, 8, 16)


def _build_base_load_features(
    current_timestep: int,
    current_base_load: float,
    base_load_history: list[float], # dict is converted to list for calculations
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

    lags: dict[int, float] = {}
    lag_pads: dict[int, int] = {}
    for step in LAG_STEPS:
        lags[step], lag_pads[step] = _lag(step)

    # direct x-step change; 0.0 if the underlying lag is padded
    deltas = {
        step: (current_base_load - lags[step] if lag_pads[step] == 0 else 0.0)
        for step in DELTA_STEPS
    }
    # deviation of the x-step change from the most recent 1-step change
    accs = {step: deltas[step] - deltas[1] for step in ACC_STEPS}

    time_sin, time_cos = encode_time_cyclic(current_timestep)

    features = {
        "timestep": current_timestep,
        "base_load": current_base_load,
        "time_sin": time_sin,
        "time_cos": time_cos,
    }

    for step in LAG_STEPS:
        features[f"base_load_lag_{step}"] = lags[step]
        features[f"base_load_lag_{step}_is_pad"] = lag_pads[step]

    for window in MA_WINDOWS:
        features[f"base_load_ma_{window}"] = _rolling_mean(window)

    for window in STD_WINDOWS:
        features[f"base_load_std_{window}"] = _rolling_std(window)

    for step in DELTA_STEPS:
        features[f"base_load_delta_{step}"] = deltas[step]

    for step in ACC_STEPS:
        features[f"base_load_acc_{step}"] = accs[step]

    if round_values:
        for key, value in list(features.items()):
            if isinstance(value, float):
                features[key] = round(value, 3)

    return features


def _try_bypass(current_timestep: int) -> float | None:
    if current_timestep >= 96:
        return 0.0
    return None
