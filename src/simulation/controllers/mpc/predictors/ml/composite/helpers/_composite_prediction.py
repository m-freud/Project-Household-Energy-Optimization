from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

from src.simulation.controllers.mpc.predictors.ml.model_interface import ModelLike

Task = Literal["regression", "classification"]
FeatureBuilder = Callable[[int, float, list[float]], dict[str, Any]]


def _linear_interpolate(predictions: dict[int, float], horizon: int) -> list[float]:
    known_horizons = []
    for h in sorted(predictions):
        known_horizons.append(h)

    if not known_horizons:
        raise ValueError("At least one prediction at horizon 0 is required.")

    for left, right in zip(known_horizons, known_horizons[1:]):
        gap = right - left
        for offset in range(left + 1, right):
            predictions[offset] = predictions[left] + (
                predictions[right] - predictions[left]
            ) * (offset - left) / gap

    last = known_horizons[-1]

    if last < horizon - 1:
        for offset in range(last + 1, horizon):
            predictions[offset] = predictions[last]

    return [predictions[offset] for offset in range(horizon)]


def _first_gap(known_horizons: Sequence[int], horizon: int) -> tuple[int, int] | None:
    bounded = sorted(h for h in known_horizons if 0 <= h < horizon)
    for left, right in zip(bounded, bounded[1:]):
        if right - left > 1:
            return left, right - left
    if bounded and bounded[-1] < horizon - 1:
        return bounded[-1], horizon - bounded[-1]
    return None


def _largest_fitting_horizon(model_bank: dict[int, ModelLike], gap: int) -> int | None:
    fitting = [offset for offset in model_bank if offset <= gap]
    return max(fitting) if fitting else None


def _predict_initial(
    model_bank: dict[int, ModelLike],
    current_timestep: int,
    current_value: float,
    history: list[float],
    model_features: list[str],
    feature_builder: FeatureBuilder,
    predictions: dict[int, float],
) -> None:
    features = feature_builder(current_timestep, current_value, history)
    for horizon_offset, model in model_bank.items():
        missing = [name for name in model_features if name not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        model_input = [features[name] for name in model_features]
        predictions[horizon_offset] = float(model.predict([model_input])[0])


def _recursive_interpolate(
    model_bank: dict[int, ModelLike],
    current_timestep: int,
    history: list[float],
    predictions: dict[int, float],
    horizon: int,
    model_features: list[str],
    feature_builder: FeatureBuilder,
    task: Task,
) -> list[float]:
    while True:
        gap_info = _first_gap(sorted(predictions), horizon)
        if gap_info is None:
            break
        left, gap = gap_info
        model_horizon = _largest_fitting_horizon(model_bank, gap - 1)
        if model_horizon is None:
            break

        history_for_model = history + [
            predictions[offset] for offset in sorted(predictions) if 0 <= offset < left
        ]
        anchor = predictions[left]
        features = feature_builder(current_timestep + left, anchor, history_for_model)
        missing = [name for name in model_features if name not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        model_input = [features[name] for name in model_features]
        value = float(model_bank[model_horizon].predict([model_input])[0])
        if task == "classification":
            value = int(round(value))
        predictions[left + model_horizon] = value

    for offset in range(horizon):
        if offset not in predictions:
            previous = max(candidate for candidate in predictions if candidate < offset)
            predictions[offset] = predictions[previous]

    return [predictions[offset] for offset in range(horizon)]


def predict_composite(
    model_bank: dict[int, ModelLike],
    current_timestep: int,
    current_value: float,
    history: list[float],
    horizon: int,
    model_features: list[str],
    feature_builder: FeatureBuilder,
    interpolation: str,
    task: Task = "regression",
) -> list[float]:
    if not model_bank:
        return [current_value] * horizon
    if horizon <= 0:
        return []
    if interpolation not in {"linear", "recursive"}:
        raise ValueError(f"Unknown interpolation method: {interpolation}")
    if task == "classification" and interpolation == "linear":
        raise ValueError("Linear interpolation is not supported for classification model banks.")

    predictions: dict[int, float] = {0: current_value}
    _predict_initial(
        model_bank=model_bank,
        current_timestep=current_timestep,
        current_value=current_value,
        history=history,
        model_features=model_features,
        feature_builder=feature_builder,
        predictions=predictions,
    )

    if interpolation == "linear":
        return _linear_interpolate(predictions, horizon)
    elif interpolation == "recursive":
        return _recursive_interpolate(
            model_bank=model_bank,
            current_timestep=current_timestep,
            history=history,
            predictions=predictions,
            horizon=horizon,
            model_features=model_features,
            feature_builder=feature_builder,
            task=task,
        )

    raise NotImplementedError(f"Unhandled interpolation method: {interpolation}")
