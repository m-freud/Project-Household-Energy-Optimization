import pytest

from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig
from src.simulation.controllers.mpc.predictors.ml.shared_helpers import (
    _build_base_load_features,
    _build_ev_status_features,
    _build_pv_gen_features,
)


def _build_features_for_target(target: str, required_features: list[str] | None = None):
    if target == "base_load":
        return _build_base_load_features(
            current_timestep=48,
            current_base_load=1.25,
            base_load_history=[float(step) / 10 for step in range(1, 48)],
            required_features=required_features,
        )
    if target == "pv_gen":
        return _build_pv_gen_features(
            current_timestep=48,
            current_pv_gen=0.75,
            pv_history=[float(step) / 10 for step in range(1, 48)],
            daylight_start=32,
            daylight_end=71,
            required_features=required_features,
        )

    return _build_ev_status_features(
        {
            "timestep": 48,
            "status": 1,
            "status_history": [0] * 30 + [1] * 10 + [2] * 7,
            "start1_earliest": 32,
            "end1_latest": 50,
            "start2_earliest": 70,
            "end2_latest": 88,
            "max_commute_steps_1": 5,
            "max_commute_steps_2": 8,
        },
        required_features=required_features,
    )


@pytest.mark.parametrize(
    ("target", "model_target"),
    [
        ("base_load", "base_load"),
        ("pv_gen", "pv_gen"),
        ("ev_status", "ev_status"),
    ],
)
@pytest.mark.parametrize("family", ["xgboost", "global_xgboost", "random_forest", "ridge"])
def test_target_feature_builder_matches_requested_model_features(target, model_target, family):
    required_features = ModelConfig.MODEL_FEATURES_BY_FAMILY[family][model_target]
    emitted_features = [
        feature for feature in required_features if feature != "prediction_horizon"
    ]

    full_features = _build_features_for_target(target)
    selected_features = _build_features_for_target(target, required_features)

    assert set(selected_features) == set(emitted_features)
    assert selected_features == {
        feature: full_features[feature] for feature in emitted_features
    }


@pytest.mark.parametrize("target", ["base_load", "pv_gen"])
def test_regression_feature_builder_skips_unrequested_rolling_statistics(
    monkeypatch,
    target,
):
    import numpy as np

    def unexpected_statistic(*_args, **_kwargs):
        raise AssertionError("unrequested rolling statistic was calculated")

    monkeypatch.setattr(np, "mean", unexpected_statistic)
    monkeypatch.setattr(np, "std", unexpected_statistic)

    features = _build_features_for_target(target, ["timestep", "time_sin"])

    assert set(features) == {"timestep", "time_sin"}


def test_ev_feature_builder_skips_unrequested_history_and_time_features(monkeypatch):
    from src.simulation.controllers.mpc.predictors.ml.shared_helpers import _ev_status

    def unexpected_calculation(*_args, **_kwargs):
        raise AssertionError("unrequested EV feature calculation was performed")

    monkeypatch.setattr(_ev_status, "_get_phase_id_seq", unexpected_calculation)
    monkeypatch.setattr(
        _ev_status,
        "_get_observed_commute_boundaries",
        unexpected_calculation,
    )
    monkeypatch.setattr(_ev_status, "encode_time_cyclic", unexpected_calculation)

    features = _build_features_for_target("ev_status", ["timestep", "status"])

    assert features == {"timestep": 48, "status": 1}
