from pathlib import Path

import pandas as pd

from models import train_utils
from src.simulation._sim_utils import controller_factories
from src.simulation.controllers.mpc.predictors.ml.horizon_groups import (
    HORIZON_GROUPS,
)
from src.simulation.controllers.mpc.predictors.ml.horizon_grouped.helpers.grouped_prediction import (
    predict_horizon_grouped,
)


class GroupModel:
    def __init__(self, value: float) -> None:
        self.value = value
        self.received_horizons: list[int] = []

    def predict(self, features):
        self.received_horizons.append(int(features[0][0]))
        return [self.value]


def _read_training_data(_path: Path, training_data: pd.DataFrame) -> pd.DataFrame:
    _ = _path
    return training_data


def test_grouped_training_uses_one_model_per_horizon_range_and_feature(
    monkeypatch,
    tmp_path,
):
    training_data = pd.DataFrame(
        {
            "next_value": [1.0, 2.0],
            "profile_id": [1, 2],
            "player_id": [11, 12],
            "feature": [0.1, 0.2],
            **{
                f"y{offset}": [float(offset), -1.0]
                for offset in range(1, 96)
            },
        }
    )
    saved_models: list[tuple[Path, pd.DataFrame]] = []

    def record_fit(
        features: pd.DataFrame,
        target: pd.Series,
        model_path: Path,
        *_args,
    ) -> None:
        _ = (target, _args)
        saved_models.append((model_path, features))

    parquet_path = tmp_path / "features.parquet"
    parquet_path.touch()
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path: _read_training_data(path, training_data),
    )
    monkeypatch.setattr(train_utils, "_fit_model", record_fit)

    train_utils.train_horizon_grouped_dataset_models(
        feature_dir=tmp_path,
        file_pattern="features.parquet",
        model_dir=tmp_path / "models" / "load",
        target_column="next_value",
        learning_rate=0.1,
        max_depth=2,
        n_estimators=10,
        dataset_name="test",
    )

    assert [path.name for path, _ in saved_models] == [
        f"{group_name}.json" for group_name, _, _ in HORIZON_GROUPS
    ]
    assert all(
        path.parent.name == "horizon_grouped_multi_model"
        for path, _ in saved_models
    )
    for (group_name, first, last), (_, features) in zip(HORIZON_GROUPS, saved_models):
        assert features["prediction_horizon"].min() == first
        assert features["prediction_horizon"].max() == last
        assert set(features["prediction_horizon"]) == set(range(first, last + 1))
        assert "player_id" not in features


def test_grouped_prediction_routes_each_offset_to_its_group_model():
    models = {
        group_name: GroupModel(index)
        for index, (group_name, _, _) in enumerate(HORIZON_GROUPS, start=1)
    }
    features_by_group = {
        group_name: ["prediction_horizon"]
        for group_name, _, _ in HORIZON_GROUPS
    }

    predictions = predict_horizon_grouped(
        models_by_group=models,
        model_features_by_group=features_by_group,
        current_timestep=1,
        current_value=0.0,
        history=[],
        horizon=96,
        feature_builder=lambda *_args: {},
        task="regression",
    )

    assert len(predictions) == 96
    assert predictions[0] == 0.0
    assert predictions[1:13] == [1.0] * 12
    assert predictions[13:43] == [2.0] * 30
    assert predictions[43:96] == [3.0] * 53
    assert models["h1_12"].received_horizons == list(range(1, 13))
    assert models["h13_42"].received_horizons == list(range(13, 43))
    assert models["h43_95"].received_horizons == list(range(43, 96))


def test_grouped_prediction_respects_bypasses_across_group_boundaries():
    models = {
        group_name: GroupModel(1.0)
        for group_name, _, _ in HORIZON_GROUPS
    }
    features_by_group = {
        group_name: ["prediction_horizon"]
        for group_name, _, _ in HORIZON_GROUPS
    }

    predictions = predict_horizon_grouped(
        models_by_group=models,
        model_features_by_group=features_by_group,
        current_timestep=1,
        current_value=0.0,
        history=[],
        horizon=15,
        feature_builder=lambda *_args: {},
        task="regression",
        bypass=lambda timestep, _history, _value: (
            0.0 if timestep in {2, 13} else None
        ),
    )

    assert predictions[1:3] == [1.0, 0.0]
    assert predictions[13] == 0.0
    assert models["h1_12"].received_horizons == [1, *range(3, 13)]
    assert models["h13_42"].received_horizons == [14]


def test_grouped_predictor_builder_loads_isolated_group_models(monkeypatch):
    model_directories = {
        target: Path(target)
        for target in ("base_load", "pv_gen", "ev1_status", "ev2_status")
    }
    loaded_paths: list[Path] = []
    stub_model = GroupModel(0.0)

    def load_model(path, _model_class):
        loaded_paths.append(path)
        return stub_model

    monkeypatch.setattr(controller_factories, "_load_xgb_model", load_model)

    predictor = controller_factories.build_horizon_grouped_multi_model_predictor(
        model_directories
    )

    assert len(loaded_paths) == 4 * len(HORIZON_GROUPS)
    assert all(path.parent.name == "horizon_grouped_multi_model" for path in loaded_paths)
    assert {
        path.name for path in loaded_paths
    } == {f"{group_name}.json" for group_name, _, _ in HORIZON_GROUPS}
    assert set(predictor.base_load_model_bank) == {
        group_name for group_name, _, _ in HORIZON_GROUPS
    }

    stub_predictor = object()
    monkeypatch.setattr(
        controller_factories,
        "build_horizon_grouped_multi_model_predictor",
        lambda _directories: stub_predictor,
    )
    factories = controller_factories.build_controller_factories(
        ["mpc_horizon_grouped_multi_model"]
    )
    assert set(factories) == {"mpc_horizon_grouped_multi_model"}
