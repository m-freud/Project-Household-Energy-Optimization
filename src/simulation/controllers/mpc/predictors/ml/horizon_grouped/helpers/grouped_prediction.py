from collections.abc import Callable, Mapping
from typing import Any, Literal

from src.simulation.controllers.mpc.predictors.ml.global_model.helpers._global_model_prediction import (
    Bypass,
    _apply_forecast_bypasses,
    _model_input,
    _try_forecast_bypass,
)
from src.simulation.controllers.mpc.predictors.ml.horizon_groups import (
    HORIZON_GROUPS,
    horizon_group_name,
)
from src.simulation.controllers.mpc.predictors.ml.model_interface import (
    ModelLike,
    get_model_feature_names,
)

Task = Literal["regression", "classification"]
FeatureBuilder = Callable[[int, float, list[float]], dict[str, Any]]


def get_group_model_features(
    models_by_group: Mapping[str, ModelLike],
    fallback_features: list[str],
) -> dict[str, list[str]]:
    expected_groups = {name for name, _, _ in HORIZON_GROUPS}
    actual_groups = set(models_by_group)
    if actual_groups != expected_groups:
        raise ValueError(
            "Horizon-group model bank has invalid groups: "
            f"missing={sorted(expected_groups - actual_groups)}, "
            f"unexpected={sorted(actual_groups - expected_groups)}"
        )

    features_by_group = {
        group_name: get_model_feature_names(model, fallback_features)
        for group_name, model in models_by_group.items()
    }
    first_group = HORIZON_GROUPS[0][0]
    reference_features = features_by_group[first_group]
    mismatched_groups = [
        group_name
        for group_name, features in features_by_group.items()
        if features != reference_features
    ]
    if mismatched_groups:
        raise ValueError(
            "Horizon-group models must use the same ordered feature schema; "
            f"mismatched groups={sorted(mismatched_groups)}"
        )
    return features_by_group


def predict_horizon_grouped(
    models_by_group: Mapping[str, ModelLike],
    model_features_by_group: Mapping[str, list[str]],
    current_timestep: int,
    current_value: float,
    history: list[float],
    horizon: int,
    feature_builder: FeatureBuilder,
    task: Task,
    bypass: Bypass | None = None,
) -> list[float]:
    if horizon <= 0:
        return []

    predictions: dict[int, float] = {0: current_value}
    max_model_horizon = HORIZON_GROUPS[-1][2]
    for offset in range(1, horizon):
        bypass_value = _try_forecast_bypass(
            bypass,
            current_timestep,
            current_value,
            history,
            predictions,
            offset,
        )
        if bypass_value is not None:
            predictions[offset] = (
                int(round(bypass_value))
                if task == "classification"
                else bypass_value
            )
            continue

        if offset > max_model_horizon:
            predictions[offset] = predictions[offset - 1]
            continue

        group_name = horizon_group_name(offset)
        model = models_by_group[group_name]
        model_input = _model_input(
            model_features_by_group[group_name],
            feature_builder,
            current_timestep,
            current_value,
            history,
            offset,
        )
        value = float(model.predict([model_input])[0])  # type: ignore[index]
        predictions[offset] = (
            int(round(value))
            if task == "classification"
            else value
        )

    values = [predictions[offset] for offset in range(horizon)]
    return _apply_forecast_bypasses(
        values,
        bypass,
        current_timestep,
        history,
        task,
    )
