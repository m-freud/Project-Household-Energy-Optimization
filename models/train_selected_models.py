import argparse
import json
import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from src.simulation.controllers.mpc.predictors.ml.shared_helpers._base_load import (
    _build_base_load_features,
)
from training.tuning import pred_hypam_sweep
from training.tuning import pv_ev_hypam_sweep


ROOT_DIR = Path(__file__).resolve().parents[1]
FEATURE_ROOT = ROOT_DIR / "training" / "features" / "feature_parquets"
TUNING_ROOT = ROOT_DIR / "training" / "tuning"
MODEL_ROOT = ROOT_DIR / "models"
DIRECT_HORIZONS = (8, 16, 32, 64)
SPLITS_TO_FIT = ("train", "validation")
IDENTIFIER_COLUMNS = {"profile_id", "household_id", "date", "ev_key", "phase"}
TARGET_PATTERN = re.compile(r"y\d+")
BASE_PARAMS: dict[str, Any] = {"random_state": 42, "n_jobs": -1}
Dataset = Literal["load", "pv", "ev1", "ev2"]


def _read_load_split(split: str) -> pd.DataFrame:
    split_dir = FEATURE_ROOT / "load" / split
    files = sorted(split_dir.glob("ch_load_features_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No load feature Parquets found in {split_dir}")
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _load_feature_columns(frame: pd.DataFrame) -> list[str]:
    return pred_hypam_sweep._get_feature_columns(frame)


def _rebuild_load_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    split: str,
) -> pd.DataFrame:
    required = {"profile_id", "timestep", "base_load"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{split} load data is missing columns: {missing}")
    rebuilt = frame.copy()
    for profile_id, profile in frame.groupby("profile_id", sort=False):
        profile = profile.sort_values("timestep")
        timesteps = profile["timestep"].tolist()
        if timesteps != list(range(1, 97)):
            raise ValueError(
                f"Expected load profile {profile_id} in {split} data to have "
                "exactly timesteps 1-96"
            )
        values = profile["base_load"].to_numpy(dtype=float)
        if pd.isna(values).any():
            raise ValueError(f"Load profile {profile_id} in {split} has missing values")
        rows = []
        for index, value in enumerate(values):
            features = _build_base_load_features(
                current_timestep=index + 1,
                current_base_load=float(value),
                base_load_history=values[:index].tolist(),
                required_features=feature_columns,
            )
            missing_features = set(feature_columns) - set(features)
            if missing_features:
                raise ValueError(
                    f"Runtime load feature builder omitted {sorted(missing_features)}"
                )
            rows.append({column: features[column] for column in feature_columns})
        rebuilt.loc[profile.index, feature_columns] = pd.DataFrame(
            rows,
            index=profile.index,
        )
    if rebuilt[feature_columns].isna().any().any():
        raise ValueError(f"Rebuilt {split} load features contain missing values")
    return rebuilt


def _read_and_rebuild(
    dataset: Dataset,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    for split in SPLITS_TO_FIT:
        if dataset == "load":
            raw = _read_load_split(split)
            columns = _load_feature_columns(raw) if feature_columns is None else feature_columns
            frames.append(_rebuild_load_features(raw, columns, split))
        else:
            raw = pv_ev_hypam_sweep._read_split(dataset, split)
            columns = (
                pv_ev_hypam_sweep._get_feature_columns(raw, dataset)
                if feature_columns is None
                else feature_columns
            )
            frames.append(
                pv_ev_hypam_sweep._build_runtime_feature_frame(
                    dataset,
                    raw,
                    columns,
                    split,
                )
            )
    if feature_columns is None:
        raise RuntimeError(f"Feature columns were not resolved for {dataset}")
    return pd.concat(frames, ignore_index=True), feature_columns


def _read_best_params(dataset: Dataset, horizon: int | None) -> dict[str, Any]:
    directory = "single_step" if horizon is None else f"step_{horizon}"
    best_path = TUNING_ROOT / dataset / directory / "best_params.json"
    if not best_path.is_file():
        raise FileNotFoundError(
            f"Missing selected hyperparameters for {dataset} "
            f"{directory}: {best_path}"
        )
    with best_path.open(encoding="utf-8") as file:
        result = json.load(file)
    params = result.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError(f"Invalid parameter selection in {best_path}")
    expected_params = {"n_estimators", "max_depth", "learning_rate"}
    if set(params) != expected_params:
        raise ValueError(
            f"Expected only {sorted(expected_params)} in {best_path}, "
            f"found {sorted(params)}"
        )
    return params


def _target_column(dataset: Dataset, horizon: int | None) -> str:
    if horizon is not None:
        if dataset not in {"load", "pv"} or horizon not in DIRECT_HORIZONS:
            raise ValueError(f"Unsupported direct target: {dataset} t+{horizon}")
        return f"y{horizon}"
    return "next_value" if dataset in {"load", "pv"} else "next_state"


def _make_model(
    dataset: Dataset,
    params: dict[str, Any],
) -> XGBRegressor | XGBClassifier:
    if dataset in {"load", "pv"}:
        return XGBRegressor(
            **BASE_PARAMS,
            objective="reg:squarederror",
            **params,
        )
    return XGBClassifier(
        **BASE_PARAMS,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        **params,
    )


def _fit_and_save(
    dataset: Dataset,
    frame: pd.DataFrame,
    feature_columns: list[str],
    horizon: int | None,
    params: dict[str, Any],
) -> Path:
    target_column = _target_column(dataset, horizon)
    if target_column not in frame.columns:
        raise ValueError(f"{dataset} training data is missing '{target_column}'")
    valid = frame[target_column].ne(-1)
    selected = frame.loc[valid]
    if selected.empty:
        raise ValueError(f"No valid training rows for {dataset} {target_column}")
    inputs = selected[feature_columns]
    targets = selected[target_column]
    if inputs.isna().any().any() or targets.isna().any():
        raise ValueError(f"Missing values in {dataset} training data for {target_column}")
    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(inputs[column])
    ]
    if non_numeric:
        raise ValueError(f"Non-numeric model features for {dataset}: {non_numeric}")

    model = _make_model(dataset, params)
    model.fit(inputs, targets)
    if horizon is None:
        output_paths = [
            MODEL_ROOT / dataset / "single_step" / "single_step.json",
            MODEL_ROOT
            / dataset
            / "per_horizon_single_step"
            / "single_y1.json",
        ]
    else:
        output_paths = [
            MODEL_ROOT
            / dataset
            / "per_horizon_single_step"
            / f"single_y{horizon}.json"
        ]
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(output_path)
        print(
            f"Trained {dataset} {target_column} on {len(selected)} train+validation "
            f"rows ({len(feature_columns)} features), params={params}; "
            f"saved {output_path}"
        )
    return output_paths[0]


def train_selected_models(datasets: tuple[Dataset, ...] = ("load", "pv", "ev1", "ev2")) -> None:
    """Fit selected single-step and direct-anchor XGBoost models on train+validation."""
    for dataset in datasets:
        raw_training = (
            _read_load_split("train")
            if dataset == "load"
            else pv_ev_hypam_sweep._read_split(dataset, "train")
        )
        feature_columns = (
            _load_feature_columns(raw_training)
            if dataset == "load"
            else pv_ev_hypam_sweep._get_feature_columns(raw_training, dataset)
        )
        training_frame, feature_columns = _read_and_rebuild(dataset, feature_columns)
        horizons: tuple[int | None, ...] = (
            (None, *DIRECT_HORIZONS)
            if dataset in {"load", "pv"}
            else (None,)
        )
        for horizon in horizons:
            params = _read_best_params(dataset, horizon)
            _fit_and_save(
                dataset,
                training_frame,
                feature_columns,
                horizon,
                params,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain selected single-step and direct-anchor models using "
            "train+validation data."
        )
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=("load", "pv", "ev1", "ev2"),
        default=["load", "pv", "ev1", "ev2"],
        help="Targets to train (default: all load, PV, EV1, and EV2 models).",
    )
    args = parser.parse_args()
    train_selected_models(tuple(args.targets))


if __name__ == "__main__":
    main()
