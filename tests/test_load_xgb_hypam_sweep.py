import json

import numpy as np
import pandas as pd

from training.tuning import pred_hypam_sweep


def test_default_load_grid_has_60_parameter_combinations():
    assert len(pred_hypam_sweep.build_param_grid(pred_hypam_sweep.LOAD_GRID_1)) == 60


class BiasedRecursiveRegressor:
    def __init__(self, *, n_estimators: int, **_params):
        self.bias = n_estimators / 10

    def fit(self, features, target):
        self.fit_rows = len(features)
        return self

    def predict(self, features):
        return features["base_load"].to_numpy() + self.bias


def _write_split(feature_root, split, profile_count=2, base_load=0.0):
    split_dir = feature_root / split
    split_dir.mkdir(parents=True)
    frames = []
    for profile_index in range(profile_count):
        base_load_values = np.full(96, base_load, dtype=float)
        frame = pd.DataFrame(
            {
                "profile_id": [f"{split}-{profile_index}"] * 96,
                "household_id": [profile_index] * 96,
                "date": ["2023-01-01"] * 96,
                "timestep": range(1, 97),
                "base_load": base_load_values,
                "time_sin": np.sin(2 * np.pi * np.arange(1, 97) / 96),
            }
        )
        for offset in range(1, 96):
            frame[f"y{offset}"] = [
                *base_load_values[offset:],
                *([-1.0] * offset),
            ]
        frame["next_value"] = frame["y1"]
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(
        split_dir / "ch_load_features_1-1.parquet",
        index=False,
    )


def test_recursive_rollout_reuses_prior_predictions_and_scores_days_equally():
    frame = pd.concat(
        [
            pd.DataFrame(
                {
                    "profile_id": [f"profile-{profile_index}"] * 96,
                    "household_id": [profile_index] * 96,
                    "date": ["2023-01-01"] * 96,
                    "timestep": range(1, 97),
                    "base_load": np.zeros(96),
                    "time_sin": np.sin(2 * np.pi * np.arange(1, 97) / 96),
                }
            )
            for profile_index in range(2)
        ],
        ignore_index=True,
    )
    model = BiasedRecursiveRegressor(n_estimators=10)

    recursive_rmse, next_step_rmse, errors = (
        pred_hypam_sweep._score_recursive_full_day(
            model,
            frame,
            ["base_load", "timestep", "time_sin"],
            "validation",
        )
    )

    rollout_lengths = [96 - start for start in pred_hypam_sweep.FORECAST_START_TIMESTEPS]
    expected_rmse = np.sqrt(
        sum(
            sum(step**2 for step in range(1, rollout_length + 1))
            for rollout_length in rollout_lengths
        )
        / sum(rollout_lengths)
    )
    assert recursive_rmse == expected_rmse
    assert next_step_rmse == 1.0
    assert len(errors) == 2 * sum(rollout_lengths)
    first_profile_errors = errors.loc[errors["profile_id"] == "profile-0"]
    first_rollout = first_profile_errors.loc[
        first_profile_errors["start_timestep"] == 12
    ]
    assert first_rollout["predicted"].tolist() == list(
        map(float, range(1, 96 - 12 + 1))
    )
    assert first_rollout["timestep"].tolist() == list(range(13, 97))
    assert set(first_profile_errors["start_timestep"]) == set(
        pred_hypam_sweep.FORECAST_START_TIMESTEPS
    )


def test_sweep_uses_validation_recursive_rmse_without_reading_test(
    monkeypatch,
    tmp_path,
):
    feature_root = tmp_path / "features"
    output_dir = tmp_path / "sweep"
    _write_split(feature_root, "train")
    _write_split(feature_root, "validation")
    monkeypatch.setattr(pred_hypam_sweep, "FEATURE_DIR", feature_root)
    monkeypatch.setattr(
        pred_hypam_sweep,
        "SINGLE_STEP_OUTPUT_DIR",
        output_dir,
    )
    monkeypatch.setattr(pred_hypam_sweep, "XGBRegressor", BiasedRecursiveRegressor)

    results = pred_hypam_sweep.hypam_sweep(
        {"n_estimators": [10, 20], "max_depth": [2]}
    )

    rollout_lengths = [96 - start for start in pred_hypam_sweep.FORECAST_START_TIMESTEPS]
    expected_rmse = np.sqrt(
        sum(
            sum(step**2 for step in range(1, rollout_length + 1))
            for rollout_length in rollout_lengths
        )
        / sum(rollout_lengths)
    )
    assert results["validation_recursive_rmse"].tolist() == [
        expected_rmse,
        2 * expected_rmse,
    ]
    assert results["validation_next_step_rmse"].tolist() == [1.0, 2.0]
    assert results["forecast_start_timesteps"].tolist() == [
        "12,24,36,48,60,72,84",
        "12,24,36,48,60,72,84",
    ]
    with (output_dir / "best_params.json").open() as file:
        best = json.load(file)
    assert best == {
        "params": {"n_estimators": 10, "max_depth": 2},
        "validation_recursive_rmse": expected_rmse,
        "validation_next_step_rmse": 1.0,
        "forecast_start_timesteps": [12, 24, 36, 48, 60, 72, 84],
    }
    assert len(pd.read_csv(output_dir / "best_validation_profile_errors.csv")) == (
        2 * sum(96 - start for start in pred_hypam_sweep.FORECAST_START_TIMESTEPS)
    )
    assert not (feature_root / "test").exists()


def test_final_test_evaluation_refits_selected_model_on_train_and_validation(
    monkeypatch,
    tmp_path,
):
    feature_root = tmp_path / "features"
    output_dir = tmp_path / "sweep"
    output_dir.mkdir()
    _write_split(feature_root, "train")
    _write_split(feature_root, "validation")
    _write_split(feature_root, "test")
    with (output_dir / "best_params.json").open("w") as file:
        json.dump(
            {
                "params": {"n_estimators": 20, "max_depth": 2},
                "validation_recursive_rmse": 1.0,
                "validation_next_step_rmse": 1.0,
            },
            file,
        )
    monkeypatch.setattr(pred_hypam_sweep, "FEATURE_DIR", feature_root)
    monkeypatch.setattr(
        pred_hypam_sweep,
        "SINGLE_STEP_OUTPUT_DIR",
        output_dir,
    )
    monkeypatch.setattr(pred_hypam_sweep, "XGBRegressor", BiasedRecursiveRegressor)

    result = pred_hypam_sweep.evaluate_best_on_test()

    rollout_lengths = [96 - start for start in pred_hypam_sweep.FORECAST_START_TIMESTEPS]
    expected_rmse = 2 * np.sqrt(
        sum(
            sum(step**2 for step in range(1, rollout_length + 1))
            for rollout_length in rollout_lengths
        )
        / sum(rollout_lengths)
    )
    assert result["train_validation_rows"] == 2 * 2 * 95
    assert result["test_rows"] == 2 * 95
    assert result["test_recursive_rmse"] == expected_rmse
    assert result["test_next_step_rmse"] == 2.0
    assert result["forecast_start_timesteps"] == [12, 24, 36, 48, 60, 72, 84]
    assert len(pd.read_csv(output_dir / "test_profile_errors.csv")) == (
        2 * sum(rollout_lengths)
    )


def test_direct_horizon_sweep_uses_configured_actual_start_rows_and_own_folder(
    monkeypatch,
    tmp_path,
):
    feature_root = tmp_path / "features"
    output_root = tmp_path / "load-output"
    _write_split(feature_root, "train")
    _write_split(feature_root, "validation")
    monkeypatch.setattr(pred_hypam_sweep, "FEATURE_DIR", feature_root)
    monkeypatch.setattr(pred_hypam_sweep, "LOAD_OUTPUT_DIR", output_root)
    monkeypatch.setattr(pred_hypam_sweep, "XGBRegressor", BiasedRecursiveRegressor)

    results = pred_hypam_sweep.direct_horizon_sweep(
        8,
        {"n_estimators": [10, 20], "max_depth": [2]},
    )

    output_dir = output_root / "step_8"
    assert results["validation_direct_rmse"].tolist() == [1.0, 2.0]
    assert results["validation_pooled_rmse"].tolist() == [1.0, 2.0]
    assert results["evaluation_start_timesteps"].tolist() == [
        "12,24,36,48,60,72,84",
        "12,24,36,48,60,72,84",
    ]
    errors = pd.read_csv(output_dir / "best_validation_profile_errors.csv")
    assert set(errors["start_timestep"]) == set(
        pred_hypam_sweep.FORECAST_START_TIMESTEPS
    )
    assert (errors["target_timestep"] == errors["start_timestep"] + 8).all()
    assert len(errors) == 2 * 7
    assert not (feature_root / "test").exists()


def test_long_direct_horizon_uses_only_starts_within_the_day():
    frame = pd.DataFrame(
        {
            "profile_id": ["profile"] * 96,
            "household_id": [1] * 96,
            "date": ["2023-01-01"] * 96,
            "timestep": range(1, 97),
            "base_load": np.zeros(96),
            "feature": np.zeros(96),
        }
    )
    for offset in range(1, 96):
        frame[f"y{offset}"] = [*([0.0] * (96 - offset)), *([-1.0] * offset)]

    rmse, pooled_rmse, starts, errors = pred_hypam_sweep._score_direct_horizon(
        BiasedRecursiveRegressor(n_estimators=10),
        frame,
        ["base_load", "feature"],
        64,
        "validation",
    )

    assert rmse == 1.0
    assert pooled_rmse == 1.0
    assert starts == (12, 24)
    assert errors["target_timestep"].tolist() == [76, 88]


def test_direct_horizon_test_evaluation_uses_selected_target_and_output_folder(
    monkeypatch,
    tmp_path,
):
    feature_root = tmp_path / "features"
    output_root = tmp_path / "load-output"
    output_dir = output_root / "step_8"
    output_dir.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        _write_split(feature_root, split)
    with (output_dir / "best_params.json").open("w") as file:
        json.dump(
            {
                "params": {"n_estimators": 20, "max_depth": 2},
                "horizon": 8,
                "validation_direct_rmse": 1.0,
                "validation_pooled_rmse": 1.0,
                "evaluation_start_timesteps": [12, 24, 36, 48, 60, 72, 84],
            },
            file,
        )
    monkeypatch.setattr(pred_hypam_sweep, "FEATURE_DIR", feature_root)
    monkeypatch.setattr(pred_hypam_sweep, "LOAD_OUTPUT_DIR", output_root)
    monkeypatch.setattr(pred_hypam_sweep, "XGBRegressor", BiasedRecursiveRegressor)

    result = pred_hypam_sweep.evaluate_direct_horizon_on_test(8)

    assert result["target_column"] == "y8"
    assert result["test_direct_rmse"] == 2.0
    assert result["test_pooled_rmse"] == 2.0
    assert result["train_validation_rows"] == 4 * (96 - 8)
    assert result["test_rows"] == 2 * 7
    assert len(pd.read_csv(output_dir / "test_profile_errors.csv")) == 2 * 7
