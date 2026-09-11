from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

from src.simulation.controllers.mpc.predictors.ml.model_interface import ModelLike

Task = Literal["regression", "classification"]
FeatureBuilder = Callable[[int, float, list[float]], dict[str, Any]]


def _linear_interpolate(predictions: dict[int, float], horizon: int) -> list[float]:
    known = sorted(predictions)
    if not known:
        raise ValueError("At least one prediction is required.")
    for left, right in zip(known, known[1:]):
        gap = right - left
        for offset in range(left + 1, right):
            predictions[offset] = predictions[left] + (
                predictions[right] - predictions[left]
            ) * (offset - left) / gap
    last = known[-1]
    for offset in range(last + 1, horizon):
        predictions[offset] = predictions[last]
    return [predictions[offset] for offset in range(horizon)]


def _first_gap(known: Sequence[int], horizon: int) -> tuple[int, int] | None:
    bounded = sorted(offset for offset in known if 0 <= offset < horizon)
    for left, right in zip(bounded, bounded[1:]):
        if right - left > 1:
            return left, right - left
    if bounded and bounded[-1] < horizon - 1:
        return bounded[-1], horizon - bounded[-1]
    return None


def _largest_fitting_horizon(horizons: Sequence[int], gap: int) -> int | None:
    fitting = [offset for offset in horizons if offset <= gap]
    return max(fitting) if fitting else None


def _model_input(
    model_features: list[str],
    feature_builder: FeatureBuilder,
    timestep: int,
    value: float,
    history: list[float],
    prediction_horizon: int,
) -> list[Any]:
    features = feature_builder(timestep, value, history)
    features["prediction_horizon"] = prediction_horizon
    missing = [name for name in model_features if name not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    return [features[name] for name in model_features]


def _recursive_interpolate(
    model: ModelLike,
    horizons: Sequence[int],
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
        model_horizon = _largest_fitting_horizon(horizons, gap - 1)
        if model_horizon is None:
            break
        history_for_model = history + [
            predictions[offset] for offset in sorted(predictions) if 0 <= offset < left
        ]
        model_input = _model_input(
            model_features,
            feature_builder,
            current_timestep + left,
            predictions[left],
            history_for_model,
            model_horizon,
        )
        value = float(model.predict([model_input])[0])  # type: ignore[index]
        predictions[left + model_horizon] = int(round(value)) if task == "classification" else value

    for offset in range(horizon):
        if offset not in predictions:
            previous = max(candidate for candidate in predictions if candidate < offset)
            predictions[offset] = predictions[previous]
    return [predictions[offset] for offset in range(horizon)]


def predict_global(
    model: ModelLike,
    prediction_horizons: Sequence[int],
    current_timestep: int,
    current_value: float,
    history: list[float],
    horizon: int,
    model_features: list[str],
    feature_builder: FeatureBuilder,
    interpolation: str,
    task: Task = "regression",
) -> list[float]:
    if horizon <= 0:
        return []
    horizons = sorted(set(int(offset) for offset in prediction_horizons if int(offset) > 0))
    if not horizons:
        return [current_value] * horizon
    if interpolation not in {"linear", "recursive"}:
        raise ValueError(f"Unknown interpolation method: {interpolation}")
    if task == "classification" and interpolation == "linear":
        raise ValueError("Linear interpolation is not supported for classification model banks.")

    predictions: dict[int, float] = {0: current_value}
    for prediction_horizon in horizons:
        model_input = _model_input(
            model_features,
            feature_builder,
            current_timestep,
            current_value,
            history,
            prediction_horizon,
        )
        value = float(model.predict([model_input])[0])  # type: ignore[index]
        predictions[prediction_horizon] = int(round(value)) if task == "classification" else value

    if interpolation == "linear":
        return _linear_interpolate(predictions, horizon)
    return _recursive_interpolate(
        model=model,
        horizons=horizons,
        current_timestep=current_timestep,
        history=history,
        predictions=predictions,
        horizon=horizon,
        model_features=model_features,
        feature_builder=feature_builder,
        task=task,
    )
