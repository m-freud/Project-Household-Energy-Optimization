from typing import cast

import pytest

from src.simulation.controllers.mpc.predictors.ml.model_interface import ModelLike
from src.simulation.controllers.mpc.predictors.ml.composite.helpers._composite_prediction import (
    predict_composite,
)
from src.simulation.controllers.mpc.predictors.ml.global_model.helpers._global_model_prediction import (
    predict_global,
)


class CountingRegressor:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def predict(self, features: list[list[float]]) -> list[float]:
        self.calls += 1
        return [self.value] * len(features)


def _bypass_after_timestep_96(
    current_timestep: int,
    _history: list[float],
    _current_value: float,
) -> float | None:
    _ = _history, _current_value
    return 0.0 if current_timestep >= 96 else None


def _build_features(
    _timestep: int,
    value: float,
    _history: list[float],
) -> dict[str, float]:
    _ = _timestep, _history
    return {"value": value}


@pytest.mark.parametrize("interpolation", ["linear", "recursive"])
def test_global_model_skips_bypassed_anchor_predictions_and_bypasses_each_step(interpolation):
    model = CountingRegressor(10.0)

    prediction = predict_global(
        model=cast(ModelLike, model),
        prediction_horizons=[1, 2],
        current_timestep=95,
        current_value=5.0,
        history=[],
        horizon=4,
        model_features=["value"],
        feature_builder=_build_features,
        interpolation=interpolation,
        bypass=_bypass_after_timestep_96,
    )

    assert prediction == [5.0, 10.0, 0.0, 0.0]
    assert model.calls == 1


@pytest.mark.parametrize("interpolation", ["linear", "recursive"])
def test_composite_model_skips_bypassed_anchor_predictions_and_bypasses_each_step(interpolation):
    model = CountingRegressor(10.0)

    prediction = predict_composite(
        model_bank=cast(dict[int, ModelLike], {1: model, 2: model}),
        current_timestep=95,
        current_value=5.0,
        history=[],
        horizon=4,
        model_features=["value"],
        feature_builder=_build_features,
        interpolation=interpolation,
        bypass=_bypass_after_timestep_96,
    )

    assert prediction == [5.0, 10.0, 0.0, 0.0]
    assert model.calls == 1
