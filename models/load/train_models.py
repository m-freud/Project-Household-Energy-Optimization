from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor


MODEL_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT_DIR / "training" / "features" / "feature_parquets" / "load"
MODEL_PATH = MODEL_DIR / "single_step" / "single_step.json"
MULTI_STEP_MODEL_PATH = MODEL_DIR / "multi_step" / "multi_step.json"
PER_HORIZON_MODEL_DIR = MODEL_DIR / "per_horizon_single_step"
TARGET_COLUMN = "next_value"
MULTI_STEP_TARGET_COLUMNS = ("y1", "y2", "y4", "y8", "y16", "y32", "y64")
TARGET_COLUMNS = {
    "next_value",
    "y1",
    "y2",
    "y4",
    "y8",
    "y16",
    "y32",
    "y64",
}
IDENTIFIER_COLUMNS = {"profile_id", "household_id", "date"}
MODEL_PARAMS = {
    "learning_rate": 0.02,
    "max_depth": 4,
    "n_estimators": 200,
    "objective": "reg:squarederror",
    "random_state": 42,
}


def load_training_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    parquet_paths = sorted(FEATURE_DIR.glob("ch_load_features_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No load feature Parquets found in {FEATURE_DIR}")

    feature_frames = [pd.read_parquet(path) for path in parquet_paths]
    training_df = pd.concat(feature_frames, ignore_index=True)
    if TARGET_COLUMN not in training_df.columns:
        raise ValueError(f"Training data is missing target column '{TARGET_COLUMN}'")

    feature_columns = [
        column
        for column in training_df.columns
        if column not in IDENTIFIER_COLUMNS | TARGET_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("No model feature columns found in the load Parquets")

    valid_targets = training_df[TARGET_COLUMN].ne(-1)
    training_df = training_df.loc[valid_targets]
    if training_df.empty:
        raise ValueError("No valid next-step target rows found in the load Parquets")
    if training_df[feature_columns + [TARGET_COLUMN]].isna().any().any():
        raise ValueError("Load training data contains missing model features or targets")

    return (
        training_df[feature_columns],
        training_df[TARGET_COLUMN],
        feature_columns,
    )


def train_single_step_model() -> None:
    features, target, feature_columns = load_training_data()
    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(features, target)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    print(
        f"Trained single-step load model with {len(features)} rows "
        f"and {len(feature_columns)} features"
    )
    print(f"Saved model to {MODEL_PATH}")



def train_multi_step_model():
    parquet_paths = sorted(FEATURE_DIR.glob("ch_load_features_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No load feature Parquets found in {FEATURE_DIR}")

    training_df = pd.concat(
        [pd.read_parquet(path) for path in parquet_paths],
        ignore_index=True,
    )
    missing_targets = set(MULTI_STEP_TARGET_COLUMNS) - set(training_df.columns)
    if missing_targets:
        raise ValueError(f"Training data is missing targets: {sorted(missing_targets)}")

    feature_columns = [
        column
        for column in training_df.columns
        if column not in IDENTIFIER_COLUMNS | TARGET_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("No model feature columns found in the load Parquets")

    feature_frames = []
    target_series = []
    for target_column in MULTI_STEP_TARGET_COLUMNS:
        offset = int(target_column[1:])
        valid_rows = training_df[target_column].ne(-1)
        features = training_df.loc[valid_rows, feature_columns].copy()
        features["y_offset"] = offset
        feature_frames.append(features)
        target_series.append(training_df.loc[valid_rows, target_column])

    features = pd.concat(feature_frames, ignore_index=True)
    target = pd.concat(target_series, ignore_index=True)
    if features.empty:
        raise ValueError("No valid multi-step target rows found in the load Parquets")
    if features.isna().any().any() or target.isna().any():
        raise ValueError("Load training data contains missing model features or targets")

    model = XGBRegressor(**MODEL_PARAMS)
    model.fit(features, target)

    MULTI_STEP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MULTI_STEP_MODEL_PATH)
    print(
        f"Trained multi-step load model with {len(features)} rows "
        f"and {len(features.columns)} features"
    )
    print(f"Saved model to {MULTI_STEP_MODEL_PATH}")


def train_per_horizon_models() -> None:
    parquet_paths = sorted(FEATURE_DIR.glob("ch_load_features_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No load feature Parquets found in {FEATURE_DIR}")

    training_df = pd.concat(
        [pd.read_parquet(path) for path in parquet_paths],
        ignore_index=True,
    )
    missing_targets = set(MULTI_STEP_TARGET_COLUMNS) - set(training_df.columns)
    if missing_targets:
        raise ValueError(f"Training data is missing targets: {sorted(missing_targets)}")

    feature_columns = [
        column
        for column in training_df.columns
        if column not in IDENTIFIER_COLUMNS | TARGET_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("No model feature columns found in the load Parquets")

    PER_HORIZON_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for target_column in MULTI_STEP_TARGET_COLUMNS:
        valid_rows = training_df[target_column].ne(-1)
        horizon_df = training_df.loc[valid_rows]
        if horizon_df.empty:
            raise ValueError(f"No valid training rows found for '{target_column}'")
        if horizon_df[feature_columns + [target_column]].isna().any().any():
            raise ValueError(
                f"Load training data contains missing features or targets for "
                f"'{target_column}'"
            )

        features = horizon_df[feature_columns]
        target = horizon_df[target_column]
        model = XGBRegressor(**MODEL_PARAMS)
        model.fit(features, target)

        model_path = PER_HORIZON_MODEL_DIR / f"single_{target_column}.json"
        model.save_model(model_path)
        print(
            f"Trained {target_column} load model with {len(features)} rows "
            f"and {len(feature_columns)} features"
        )
        print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train_per_horizon_models()