from typing import cast

import pytest

from src.simulation.controllers.mpc.predictors.ml.model_interface import ModelLike
from src.simulation.controllers.mpc.predictors.ml.composite.helpers._composite_prediction import (
    predict_composite,
)
from src.simulation.controllers.mpc.predictors.ml._archive.global_model.helpers._global_model_prediction import (
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


@pytest.mark.parametrize(
    "interpolation",
    ["linear", "recursive", "anchored_recursive"],
)
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


@pytest.mark.parametrize(
    "interpolation",
    ["linear", "recursive", "anchored_recursive"],
)
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


class StepAndAnchorModel:
    def __init__(self, value=None):
        self.value = value

    def predict(self, features):
        return [
            row[0] + 1 if self.value is None else self.value
            for row in features
        ]


def _assert_anchored_recursive_prediction(prediction):
    assert prediction == [10.0, 11.0, 14.0, 17.0, 20.0, 22.5, 25.0, 27.5, 30.0]


def test_composite_anchored_recursive_corrects_recursive_rollouts_to_each_anchor():
    model_bank = {
        1: StepAndAnchorModel(),
        4: StepAndAnchorModel(20.0),
        8: StepAndAnchorModel(30.0),
    }

    prediction = predict_composite(
        model_bank=cast(dict[int, ModelLike], model_bank),
        current_timestep=10,
        current_value=10.0,
        history=[9.0],
        horizon=9,
        model_features=["value"],
        feature_builder=_build_features,
        interpolation="anchored_recursive",
    )

    _assert_anchored_recursive_prediction(prediction)


def test_global_anchored_recursive_corrects_recursive_rollouts_to_each_anchor():
    class GlobalStepAndAnchorModel:
        def predict(self, features):
            return [
                row[0] + 1 if row[1] == 1 else {4: 20.0, 8: 30.0}[int(row[1])]
                for row in features
            ]

    prediction = predict_global(
        model=cast(ModelLike, GlobalStepAndAnchorModel()),
        prediction_horizons=[1, 4, 8],
        current_timestep=10,
        current_value=10.0,
        history=[9.0],
        horizon=9,
        model_features=["value", "prediction_horizon"],
        feature_builder=_build_features,
        interpolation="anchored_recursive",
    )

    _assert_anchored_recursive_prediction(prediction)


@pytest.mark.parametrize("predictor", ["composite", "global"])
def test_anchored_recursive_interpolation_rejects_classification(predictor):
    with pytest.raises(ValueError, match="classification"):
        if predictor == "composite":
            predict_composite(
                model_bank=cast(dict[int, ModelLike], {1: StepAndAnchorModel()}),
                current_timestep=10,
                current_value=0,
                history=[],
                horizon=2,
                model_features=["value"],
                feature_builder=_build_features,
                interpolation="anchored_recursive",
                task="classification",
            )
        else:
            predict_global(
                model=cast(ModelLike, StepAndAnchorModel()),
                prediction_horizons=[1],
                current_timestep=10,
                current_value=0,
                history=[],
                horizon=2,
                model_features=["value", "prediction_horizon"],
                feature_builder=_build_features,
                interpolation="anchored_recursive",
                task="classification",
            )


def test_composite_anchored_recursive_requires_single_step_model():
    with pytest.raises(ValueError, match="single-step model"):
        predict_composite(
            model_bank=cast(dict[int, ModelLike], {4: StepAndAnchorModel(20.0)}),
            current_timestep=10,
            current_value=10.0,
            history=[],
            horizon=5,
            model_features=["value"],
            feature_builder=_build_features,
            interpolation="anchored_recursive",
        )
