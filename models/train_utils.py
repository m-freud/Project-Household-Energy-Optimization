from pathlib import Path
import re
from typing import Literal

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor


IDENTIFIER_COLUMNS = {
    "profile_id",
    "household_id",
    "date",
    "player_id",
    "ev_key",
    "phase",
}
MODEL_PARAMS = {
    "random_state": 42,
}


def train_dataset_models(
    feature_dir: Path,
    file_pattern: str,
    model_dir: Path,
    target_column: str,
    learning_rate: float,
    max_depth: int,
    n_estimators: int,
    dataset_name: str,
    model_type: Literal["regression", "classification"] = "regression",
) -> None:
    if model_type not in {"regression", "classification"}:
        raise ValueError(f"Unsupported model type: {model_type}")

    parquet_paths = sorted(feature_dir.glob(file_pattern))
    if not parquet_paths:
        raise FileNotFoundError(f"No feature Parquets found in {feature_dir}")

    training_df = pd.concat(
        [pd.read_parquet(path) for path in parquet_paths],
        ignore_index=True,
    )
    if target_column not in training_df.columns:
        raise ValueError(f"Training data is missing target column '{target_column}'")

    target_columns = sorted(
        (
            column
            for column in training_df.columns
            if re.fullmatch(r"y\d+", column)
        ),
        key=lambda column: int(column[1:]),
    )
    if not target_columns or "y1" not in target_columns:
        raise ValueError("Training data must contain at least the y1 target")

    numeric_columns = training_df.select_dtypes(include="number").columns
    feature_columns = [
        column
        for column in numeric_columns
        if column not in IDENTIFIER_COLUMNS
        and column != target_column
        and column not in target_columns
    ]
    if not feature_columns:
        raise ValueError("No numeric model feature columns found in the Parquets")

    params = {
        **MODEL_PARAMS,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "n_estimators": n_estimators,
    }
    if model_type == "regression":
        params["objective"] = "reg:squarederror"
    else:
        params.update(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
        )

    single_rows = training_df[target_column].ne(-1)
    _fit_model(
        training_df.loc[single_rows, feature_columns],
        training_df.loc[single_rows, target_column],
        model_dir / "single_step" / "single_step.json",
        params,
        f"{dataset_name} single-step",
        model_type,
    )

    multi_feature_frames = []
    multi_target_series = []
    for target_name in target_columns:
        valid_rows = training_df[target_name].ne(-1)
        features = training_df.loc[valid_rows, feature_columns].copy()
        features["y_offset"] = int(target_name[1:])
        multi_feature_frames.append(features)
        multi_target_series.append(training_df.loc[valid_rows, target_name])

    _fit_model(
        pd.concat(multi_feature_frames, ignore_index=True),
        pd.concat(multi_target_series, ignore_index=True),
        model_dir / "multi_step" / "multi_step.json",
        params,
        f"{dataset_name} multi-step",
        model_type,
    )

    horizon_dir = model_dir / "per_horizon_single_step"
    for target_name in target_columns:
        valid_rows = training_df[target_name].ne(-1)
        _fit_model(
            training_df.loc[valid_rows, feature_columns],
            training_df.loc[valid_rows, target_name],
            horizon_dir / f"single_{target_name}.json",
            params,
            f"{dataset_name} {target_name}",
            model_type,
        )


def _fit_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_path: Path,
    params: dict[str, object],
    model_name: str,
    model_type: Literal["regression", "classification"],
) -> None:
    if features.empty:
        raise ValueError(f"No valid training rows found for {model_name}")
    if features.isna().any().any() or target.isna().any():
        raise ValueError(f"Training data contains missing values for {model_name}")

    if model_type == "classification":
        model = XGBClassifier(**params)
    else:
        model = XGBRegressor(**params)
    model.fit(features, target)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    print(
        f"Trained {model_name} with {len(features)} rows "
        f"and {len(features.columns)} features; saved to {model_path}"
    )
