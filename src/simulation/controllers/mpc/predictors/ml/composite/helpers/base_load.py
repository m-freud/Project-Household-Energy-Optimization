from typing import Any, cast
from pathlib import Path
import sys
import random

import matplotlib.pyplot as plt

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig # noqa
from src.simulation.controllers.mpc.predictors.ml.model_interface import TRegressor # noqa
from src.simulation.controllers.mpc.predictors.shared.make_band import make_band # noqa
from src.simulation.household import Household # noqa
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._base_load import _build_base_load_features # noqa

def _linear_interpolate(
        history: list[float],
        pred_dict: dict[int, float],
        horizon: int
        ) -> list[float]:
    known_horizons = sorted(pred_dict.keys())

    for h, next_h in zip(known_horizons, known_horizons[1:]):
        gap = next_h - h
        for missing_h in range(h + 1, next_h):
            pred_dict[missing_h] = pred_dict[h] + (pred_dict[next_h] - pred_dict[h]) * (missing_h - h) / gap

    # fill in any missing horizons at the end
    last_h = known_horizons[-1]
    for missing_h in range(last_h + 1, horizon):
        pred_dict[missing_h] = pred_dict[last_h]

    pred_list = [pred_dict[h] for h in range(horizon)]

    return pred_list


def _get_first_h_gap(known_horizons, horizon):
    bounded_horizons = sorted(h for h in known_horizons if 0 <= h < horizon)
    for prev_h, next_h in zip(bounded_horizons, bounded_horizons[1:]):
        if next_h - prev_h > 1:
            return prev_h, next_h - prev_h

    if bounded_horizons:
        last_h = bounded_horizons[-1]
        if last_h < horizon - 1:
            return last_h, horizon - last_h

    return None, None


def _get_longest_model_horizon(model_bank: dict[int, TRegressor], gap_len: int) -> int | None:
    suitable_horizons = [h for h in model_bank.keys() if h <= gap_len]
    if not suitable_horizons:
        return None
    return max(suitable_horizons)


def _recursive_interpolate(
        model_bank: dict[int, TRegressor],
        history: list[float],
        pred_dict: dict[int, float],
        horizon: int,
        current_timestep: int,
        ) -> list[float]:
    known_horizons = sorted(pred_dict.keys())
    model_features = ModelConfig.MODEL_FEATURES_BY_FAMILY["xgboost"]["base_load"]

    while True:
        first_missing_h, gap_len = _get_first_h_gap(known_horizons, horizon)
        if first_missing_h is None or gap_len is None:
            break

        model_horizon = _get_longest_model_horizon(model_bank, gap_len - 1)
        if model_horizon is None:
            break

        target_h = first_missing_h + model_horizon
        history_for_model = history + [
            pred_dict[h]
            for h in sorted(pred_dict.keys())
            if 0 <= h < first_missing_h
        ]
        model_input_features = _build_base_load_features(
            current_timestep=current_timestep + first_missing_h,
            current_base_load=pred_dict[first_missing_h],
            base_load_history=history_for_model,
        )

        for feature in model_features:
            if feature not in model_input_features:
                raise ValueError(f"Missing required feature: {feature}")

        model_input = [model_input_features[feature] for feature in model_features]
        pred_dict[target_h] = float(model_bank[model_horizon].predict([model_input])[0])  # type: ignore
        known_horizons = sorted(pred_dict.keys())

    for missing_h in range(horizon):
        if missing_h not in pred_dict:
            previous_h = max(h for h in pred_dict if h < missing_h)
            pred_dict[missing_h] = pred_dict[previous_h]

    pred_list = [pred_dict[h] for h in range(horizon)]

    return pred_list


def _interpolate_prediction(
        model_bank,
        history: list[float],
        pred_dict: dict[int, float],
        horizon: int,
        current_timestep: int,
        interpolation: str = "linear"
) -> list[float]:

    if interpolation == "linear":
        interpolated_pred = _linear_interpolate(history, pred_dict, horizon)
        return interpolated_pred

    if interpolation == "recursive":
        return _recursive_interpolate(model_bank, history, pred_dict, horizon, current_timestep)

    raise ValueError(f"Unknown interpolation method: {interpolation}")


def _predict_base_load(
        model_bank:dict[int, TRegressor],
        household:Household,
        horizon:int,
        interpolation: str = "linear"
    ) -> list[float]:
    # current values
    current_timestep = household.current_timestep
    current_base_load = household.base_load

    # init sim history
    # we use a list here for convenience
    base_load_history = list(household.history["base_load"].values())

    # anchor at horizon 0 = current (observed) value, so interpolation has a start point
    base_load_pred_dict: dict[int, float] = {0: current_base_load}

    # features are built once from the current state; direct multi-horizon
    # models all predict from the same "now", just at different target offsets
    all_features = _build_base_load_features(
        current_timestep=current_timestep,
        current_base_load=current_base_load,
        base_load_history=base_load_history,
    )

    model_features = ModelConfig.MODEL_FEATURES_BY_FAMILY["xgboost"]["base_load"]

    for h, model in model_bank.items():
        # ensure completeness of features for the model (redundant but whatever)
        for f in model_features:
            if f not in all_features:
                raise ValueError(f"Missing required feature: {f}")

        # ensure correct order of features (as in model_features)
        model_input = [all_features[f] for f in model_features]

        base_load_pred_dict[h] = float(model.predict([model_input])[0])  # type: ignore

    # now interpolate
    base_load_pred_list = _interpolate_prediction(
        model_bank=model_bank,
        history=base_load_history,
        pred_dict=base_load_pred_dict,
        horizon=horizon,
        current_timestep=current_timestep,
        interpolation=interpolation
    )

    return base_load_pred_list


def predict_base_load(
    model_bank: dict[int, TRegressor],
    household: Household,
    horizon: int=96,
    interpolation: str = "linear",
    interval_width_frct: float = 0.0,
) -> dict[str, list[float]]:
    base_load = _predict_base_load(
        model_bank=model_bank,
        household=household,
        horizon=horizon,
        interpolation=interpolation,
    )
    base_load_lb, base_load_ub = make_band(base_load, interval_width_frct)

    return {
        "base_load": base_load,
        "base_load_lb": base_load_lb,
        "base_load_ub": base_load_ub,
    }


if __name__ == "__main__":
    class _DebugConstantRegressor:
        def __init__(self, value: float):
            self.value = value

        def predict(self, model_input):
            return [self.value] * len(model_input) # * random.choice([-1, 0, 1])] * len(model_input)


    def _debug_run_base_load_interpolation() -> None:
        debug_household = Household(player_id=1, start_time=1)
        debug_household.current_timestep = 20
        debug_household.base_load = 10.0
        debug_household.history["base_load"] = {timestep: 10.0 for timestep in range(1, 20)}

        debug_model_bank = cast(
            Any,
            {horizon: _DebugConstantRegressor(horizon) for horizon in (1, 2, 4, 8, 16, 32, 64)},
        )

        for interpolation in ("linear", "recursive"):
            prediction_horizon = 96
            base_load_history = list(debug_household.history["base_load"].values())
            all_features = _build_base_load_features(
                current_timestep=debug_household.current_timestep,
                current_base_load=debug_household.base_load,
                base_load_history=base_load_history,
            )
            model_features = ModelConfig.MODEL_FEATURES_BY_FAMILY["xgboost"]["base_load"]
            initial_predictions = {0: debug_household.base_load}
            for horizon, model in debug_model_bank.items():
                model_input = [all_features[feature] for feature in model_features]
                initial_predictions[horizon] = float(model.predict([model_input])[0])

            prediction_dict = dict(initial_predictions)
            prediction = _interpolate_prediction(
                model_bank=debug_model_bank,
                history=base_load_history,
                pred_dict=prediction_dict,
                horizon=prediction_horizon,
                current_timestep=debug_household.current_timestep,
                interpolation=interpolation,
            )
            interpolated_horizons = sorted(set(prediction_dict) - set(initial_predictions))

            plt.figure(figsize=(12, 5))
            plt.plot(range(prediction_horizon), prediction, color="red", linewidth=1, label="full prediction")
            initial_horizons = sorted(initial_predictions)
            plt.scatter(
                initial_horizons,
                [initial_predictions[horizon] for horizon in initial_horizons],
                color="blue",
                zorder=3,
                label="initial model predictions",
            )
            plt.scatter(
                interpolated_horizons,
                [prediction_dict[horizon] for horizon in interpolated_horizons],
                color="red",
                zorder=3,
                label="interpolated / recursive predictions",
            )
            plt.title(f"Base-load composite prediction: {interpolation}")
            plt.xlabel("Forecast horizon")
            plt.ylabel("Base load")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

    _debug_run_base_load_interpolation()