import numpy as np
import pandas as pd
from types import SimpleNamespace

from src.simulation.controllers.mpc.predictors.ml.recursive.helpers import (
    ev_status as recursive_ev_status,
)
from training.tuning import pv_ev_hypam_sweep


class ConstantClassifier:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class

    def predict(self, features):
        return np.full(len(features), self.predicted_class)


class ConstantRegressor:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, features):
        return np.full(len(features), self.prediction)


def _ev_profile_frame():
    statuses = np.tile([0, 1, 2], 32)
    return pd.DataFrame(
        {
            "profile_id": ["ev-profile"] * 96,
            "household_id": ["ev-household"] * 96,
            "timestep": range(1, 97),
            "status": statuses,
            "phase": ["at_home"] * 96,
            "ev_key": ["ev1"] * 96,
        }
    )


def test_ev_recursive_rollout_uses_runtime_features_and_class_labels(monkeypatch):
    frame = _ev_profile_frame()
    monkeypatch.setattr(pv_ev_hypam_sweep, "_runtime_bypass", lambda *_args: None)
    monkeypatch.setattr(
        pv_ev_hypam_sweep,
        "_runtime_features",
        lambda _target, state, timestep, _columns: {
            "timestep": timestep,
            "status": state["current"],
        },
    )

    metrics, next_step_metrics, errors = (
        pv_ev_hypam_sweep._score_recursive_single_step(
            ConstantClassifier(2),
            frame,
            ["timestep", "status"],
            "ev1",
            "validation",
        )
    )

    expected_forecast_count = sum(
        96 - start for start in pv_ev_hypam_sweep.FORECAST_START_TIMESTEPS
    )
    assert len(errors) == expected_forecast_count
    assert set(errors["start_timestep"]) == set(
        pv_ev_hypam_sweep.FORECAST_START_TIMESTEPS
    )
    assert set(errors["predicted"]) == {2}
    assert metrics["accuracy"] == (errors["actual"] == 2).mean()
    assert next_step_metrics["next_step_accuracy"] == (
        errors.loc[errors["timestep"] == errors["start_timestep"] + 1, "actual"]
        == 2
    ).mean()


def test_pv_direct_scoring_uses_only_valid_forecast_starts():
    frame = pd.DataFrame(
        {
            "profile_id": ["pv-profile"] * 96,
            "household_id": ["pv-household"] * 96,
            "date": ["2023-01-01"] * 96,
            "timestep": range(1, 97),
            "pv_gen": np.arange(96, dtype=float),
            "feature": np.arange(96, dtype=float),
        }
    )
    for offset in pv_ev_hypam_sweep.DIRECT_HORIZONS:
        frame[f"y{offset}"] = [
            *np.arange(offset, 96, dtype=float),
            *([-1.0] * offset),
        ]

    metrics, starts, errors = pv_ev_hypam_sweep._score_direct_pv(
        ConstantRegressor(0.0),
        frame,
        ["timestep", "feature"],
        64,
        "validation",
    )

    assert starts == (12, 24)
    assert errors["target_timestep"].tolist() == [76, 88]
    assert metrics["direct_rmse"] == metrics["pooled_rmse"]


def test_ev_feature_selection_omits_string_identifiers_and_targets():
    frame = pd.DataFrame(
        {
            "profile_id": ["profile"],
            "household_id": ["household"],
            "ev_key": ["ev1"],
            "phase": ["at_home"],
            "timestep": [1],
            "status": [0],
            "phase_id": [0],
            "next_state": [1],
            "y1": [1],
        }
    )

    assert pv_ev_hypam_sweep._get_feature_columns(frame, "ev1") == [
        "timestep",
        "status",
        "phase_id",
    ]


def test_ev_training_features_are_rebuilt_from_observed_history():
    statuses = np.array([0] * 40 + [1] * 2 + [2] * 16 + [1] * 2 + [0] * 36)
    frame = pd.DataFrame(
        {
            "profile_id": ["ev-profile"] * 96,
            "timestep": range(1, 97),
            "status": statuses,
            "end2": [-1] * 96,
        }
    )

    rebuilt = pv_ev_hypam_sweep._build_runtime_feature_frame(
        "ev2",
        frame,
        ["timestep", "status", "end2"],
        "train",
    )

    assert rebuilt.loc[rebuilt["timestep"] == 72, "end2"].item() != -1


def test_ev_runtime_features_do_not_duplicate_current_status_in_history(monkeypatch):
    seen_histories = []
    monkeypatch.setattr(
        recursive_ev_status.ModelConfig,
        "get_model_family_name",
        lambda _model: "test",
    )
    monkeypatch.setitem(
        recursive_ev_status.ModelConfig.MODEL_FEATURES_BY_FAMILY,
        "test",
        {"ev_status": ["feature"]},
    )
    monkeypatch.setattr(
        recursive_ev_status,
        "get_model_feature_names",
        lambda _model, features: list(features),
    )
    monkeypatch.setattr(
        recursive_ev_status,
        "_build_ev_status_features",
        lambda data, required_features: (
            seen_histories.append(list(data["status_history"])) or {"feature": 1}
        ),
    )
    monkeypatch.setattr(recursive_ev_status, "_try_bypass", lambda **_kwargs: None)

    household = SimpleNamespace(
        current_timestep=3,
        ev1_at_home=True,
        ev1_at_charging_station=False,
        history={
            "ev1_at_home": {1: True, 2: True},
            "ev1_at_charging_station": {1: False, 2: False},
        },
    )

    recursive_ev_status._predict_single_ev_status(
        ConstantClassifier(0),
        household,
        "ev1",
        horizon=2,
    )

    assert seen_histories == [[0, 0]]
