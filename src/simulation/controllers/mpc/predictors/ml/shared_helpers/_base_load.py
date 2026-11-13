import numpy as np
from collections.abc import Collection

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
    required_features: Collection[str] | None = None,
) -> dict:
    requested = set(required_features) if required_features is not None else None

    def needs(name: str) -> bool:
        return requested is None or name in requested

    base_load_seq = base_load_history + [current_base_load]

    def _lag(lag: int) -> tuple[float, int]:
        idx = len(base_load_seq) - 1 - lag
        if idx >= 0:
            return float(base_load_seq[idx]), 0
        return -1.0, 1

    needed_lags: set[int] = set()
    needed_deltas: set[int] = set()
    needed_accs: set[int] = set()
    for step in LAG_STEPS:
        if needs(f"base_load_lag_{step}") or needs(f"base_load_lag_{step}_is_pad"):
            needed_lags.add(step)
    for step in DELTA_STEPS:
        if needs(f"base_load_delta_{step}"):
            needed_deltas.add(step)
    for step in ACC_STEPS:
        if needs(f"base_load_acc_{step}"):
            needed_accs.add(step)
            needed_deltas.update((step, 1))
    if needs("base_load_accel"):
        needed_deltas.update((1, 2))
    for step in needed_deltas:
        needed_lags.add(1)
        needed_lags.add(2 if step == 2 else step)

    lags: dict[int, float] = {}
    lag_pads: dict[int, int] = {}
    for step in needed_lags:
        lags[step], lag_pads[step] = _lag(step)

    deltas: dict[int, float] = {}
    for step in needed_deltas:
        if step == 2:
            deltas[step] = (
                lags[1] - lags[2]
                if lag_pads[1] == 0 and lag_pads[2] == 0
                else 0.0
            )
        else:
            deltas[step] = (
                current_base_load - lags[step]
                if lag_pads[step] == 0
                else 0.0
            )

    features: dict[str, float | int] = {}
    if needs("timestep"):
        features["timestep"] = current_timestep
    if needs("base_load"):
        features["base_load"] = current_base_load
    if needs("time_sin") or needs("time_cos"):
        time_sin, time_cos = encode_time_cyclic(current_timestep)
        if needs("time_sin"):
            features["time_sin"] = time_sin
        if needs("time_cos"):
            features["time_cos"] = time_cos

    for step in needed_lags:
        if needs(f"base_load_lag_{step}"):
            features[f"base_load_lag_{step}"] = lags[step]
        if needs(f"base_load_lag_{step}_is_pad"):
            features[f"base_load_lag_{step}_is_pad"] = lag_pads[step]

    for window in MA_WINDOWS:
        if needs(f"base_load_ma_{window}"):
            features[f"base_load_ma_{window}"] = float(
                np.mean(np.asarray(base_load_seq[-window:], dtype=float))
            )

    for window in STD_WINDOWS:
        if needs(f"base_load_std_{window}"):
            features[f"base_load_std_{window}"] = float(
                np.std(np.asarray(base_load_seq[-window:], dtype=float), ddof=0)
            )

    for step in DELTA_STEPS:
        if needs(f"base_load_delta_{step}"):
            features[f"base_load_delta_{step}"] = deltas[step]

    for step in ACC_STEPS:
        if needs(f"base_load_acc_{step}"):
            features[f"base_load_acc_{step}"] = deltas[step] - deltas[1]
    if needs("base_load_accel"):
        features["base_load_accel"] = deltas[1] - deltas[2]

    if round_values:
        for key, value in list(features.items()):
            if isinstance(value, float):
                features[key] = round(value, 3)

    return features


def _try_bypass(current_timestep: int) -> float | None:
    if current_timestep >= 96:
        return 0.0
    return None
