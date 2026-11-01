from pathlib import Path

import pandas as pd

from models import train_utils
from models.load import train_models as load_train_models


def _read_training_data(_path: Path, training_data: pd.DataFrame) -> pd.DataFrame:
    _ = _path
    return training_data


def test_generic_high_resolution_training_uses_isolated_artifact_paths(
    monkeypatch,
    tmp_path,
):
    training_data = pd.DataFrame(
        {
            "next_value": [1.0, 2.0],
            "timestep": [1, 2],
            **{
                f"y{offset}": [float(offset), -1.0]
                for offset in range(1, 96)
            },
        }
    )
    saved_paths: list[Path] = []

    (tmp_path / "features.parquet").touch()
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda path: _read_training_data(path, training_data),
    )

    def record_fit(
        features: pd.DataFrame,
        target: pd.Series,
        model_path: Path,
        *_args,
    ) -> None:
        _ = (features, target, _args)
        saved_paths.append(model_path)

    monkeypatch.setattr(train_utils, "_fit_model", record_fit)

    train_utils.train_dataset_models(
        feature_dir=tmp_path,
        file_pattern="*.parquet",
        model_dir=tmp_path / "models" / "load",
        target_column="next_value",
        learning_rate=0.1,
        max_depth=2,
        n_estimators=10,
        dataset_name="test",
        high_resolution=True,
    )

    assert saved_paths[0] == (
        tmp_path / "models" / "load" / "multi_step" / "multi_step_(1-95).json"
    )
    horizon_paths = saved_paths[1:]
    assert len(horizon_paths) == 95
    assert [path.name for path in horizon_paths] == [
        f"single_y{offset}_high_res.json" for offset in range(1, 96)
    ]
    assert all(path.parent.name == "per_horizon_single_step" for path in horizon_paths)


def test_load_training_preserves_legacy_targets_and_adds_high_resolution_paths():
    assert load_train_models.MULTI_STEP_TARGET_COLUMNS == (
        "y1",
        "y2",
        "y4",
        "y8",
        "y16",
        "y32",
        "y64",
    )
    assert load_train_models.HIGH_RES_TARGET_COLUMNS == tuple(
        f"y{offset}" for offset in range(1, 96)
    )
    assert load_train_models.MULTI_STEP_MODEL_PATH.name == "multi_step.json"
    assert load_train_models.HIGH_RES_MULTI_STEP_MODEL_PATH.name == "multi_step_(1-95).json"
