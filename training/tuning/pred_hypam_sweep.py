import argparse
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from xgboost import XGBRegressor

from src.simulation.controllers.mpc.predictors.ml.shared_helpers._base_load import (
    _build_base_load_features,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT_DIR / "training" / "features" / "feature_parquets" / "load"
OUTPUT_DIR = Path(__file__).resolve().parent / "load_single_step"
TARGET_COLUMN = "next_value"
IDENTIFIER_COLUMNS = {"profile_id", "household_id", "date"}
TARGET_PATTERN = re.compile(r"y\d+")
BASE_MODEL_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}
LOAD_GRID_1 = {
    "n_estimators": [100, 200, 300],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
}
FORECAST_START_TIMESTEPS = tuple(range(12, 85, 12))


def build_param_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid or any(not values for values in grid.values()):
        raise ValueError("The hyperparameter grid and each parameter value list must be non-empty")
    keys = list(grid)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def _read_split(split: str) -> pd.DataFrame:
    split_dir = FEATURE_DIR / split
    parquet_paths = sorted(split_dir.glob("ch_load_features_*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No load feature Parquets found in {split_dir}")

    frame = pd.concat(
        [pd.read_parquet(path) for path in parquet_paths],
        ignore_index=True,
    )
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"{split} data is missing target column '{TARGET_COLUMN}'")
    return frame


def _get_feature_columns(training_frame: pd.DataFrame) -> list[str]:
    feature_columns = [
        column
        for column in training_frame.columns
        if column not in IDENTIFIER_COLUMNS
        and column != TARGET_COLUMN
        and not TARGET_PATTERN.fullmatch(column)
    ]
    if not feature_columns:
        raise ValueError("No model feature columns found in the load training Parquets")
    non_numeric_columns = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(training_frame[column])
    ]
    if non_numeric_columns:
        raise ValueError(f"Load model features must be numeric: {non_numeric_columns}")
    return feature_columns


def _get_valid_data(
    frame: pd.DataFrame,
    feature_columns: list[str],
    split: str,
) -> tuple[pd.DataFrame, pd.Series]:
    missing_columns = [
        column
        for column in [*feature_columns, TARGET_COLUMN]
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"{split} data is missing columns: {missing_columns}")

    valid_rows = frame[TARGET_COLUMN].ne(-1)
    valid_frame = frame.loc[valid_rows]
    if valid_frame.empty:
        raise ValueError(f"No valid '{TARGET_COLUMN}' rows found in {split} data")
    if valid_frame[[*feature_columns, TARGET_COLUMN]].isna().any().any():
        raise ValueError(f"Load {split} data contains missing features or targets")
    non_numeric_columns = [
        column
        for column in [*feature_columns, TARGET_COLUMN]
        if not pd.api.types.is_numeric_dtype(valid_frame[column])
    ]
    if non_numeric_columns:
        raise ValueError(f"Load {split} data columns must be numeric: {non_numeric_columns}")
    return valid_frame[feature_columns], valid_frame[TARGET_COLUMN]


def _score_recursive_full_day(
    model: XGBRegressor,
    frame: pd.DataFrame,
    feature_columns: list[str],
    split: str,
) -> tuple[float, float, pd.DataFrame]:
    required_columns = {"profile_id", "household_id", "date", "timestep", "base_load"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{split} data is missing profile columns: {missing_columns}")

    profile_states = []
    for profile_id, profile in frame.groupby("profile_id", sort=False):
        profile = profile.sort_values("timestep")
        if profile["timestep"].tolist() != list(range(1, 97)):
            raise ValueError(
                f"Expected profile {profile_id} in {split} data to contain "
                "exactly timesteps 1-96"
            )
        actual = profile["base_load"].to_numpy(dtype=float)
        if pd.isna(actual).any():
            raise ValueError(f"Profile {profile_id} in {split} data contains missing load")
        for start_timestep in FORECAST_START_TIMESTEPS:
            profile_states.append(
                {
                    "profile_id": profile_id,
                    "household_id": profile["household_id"].iloc[0],
                    "date": profile["date"].iloc[0],
                    "actual": actual,
                    "start_timestep": start_timestep,
                    "history": actual[: start_timestep - 1].tolist(),
                    "current": float(actual[start_timestep - 1]),
                    "expected_start_features": profile.iloc[
                        start_timestep - 1
                    ][feature_columns].to_dict(),
                    "squared_errors": [],
                    "next_step_error": None,
                    "errors": [],
                }
            )

    if not profile_states:
        raise ValueError(f"No profile-days found in {split} data")

    for current_timestep in range(1, 96):
        active_states = [
            state
            for state in profile_states
            if state["start_timestep"] <= current_timestep
            and len(state["squared_errors"]) < 96 - state["start_timestep"]
        ]
        if not active_states:
            continue
        feature_rows = []
        for state in active_states:
            current_features = _build_base_load_features(
                current_timestep=current_timestep,
                current_base_load=state["current"],
                base_load_history=state["history"],
                required_features=feature_columns,
            )
            if current_timestep == state["start_timestep"]:
                for column, expected_value in state["expected_start_features"].items():
                    if not math.isclose(
                        current_features[column],
                        expected_value,
                        rel_tol=0.0,
                        abs_tol=0.00051,
                    ):
                        raise ValueError(
                            f"Recursive start features do not match {split} "
                            f"Parquet for profile {state['profile_id']} "
                            f"at timestep {current_timestep}: {column}"
                        )
            feature_rows.append(current_features)
        model_input = pd.DataFrame(feature_rows, columns=feature_columns)
        missing_features = model_input.columns[model_input.isna().any()].tolist()
        if missing_features:
            raise ValueError(
                f"Generated recursive features contain missing values: {missing_features}"
            )
        predictions = model.predict(model_input)

        for state, prediction in zip(active_states, predictions):
            prediction = float(prediction)
            actual_value = float(state["actual"][current_timestep])
            error = prediction - actual_value
            state["squared_errors"].append(error**2)
            if current_timestep == state["start_timestep"]:
                state["next_step_error"] = error
            state["errors"].append(
                {
                    "profile_id": state["profile_id"],
                    "household_id": state["household_id"],
                    "date": state["date"],
                    "start_timestep": state["start_timestep"],
                    "timestep": current_timestep + 1,
                    "actual": actual_value,
                    "predicted": prediction,
                    "error": error,
                }
            )
            state["history"].append(state["current"])
            state["current"] = prediction

    profile_states_by_profile: dict[str, list[dict[str, Any]]] = {}
    for state in profile_states:
        profile_states_by_profile.setdefault(state["profile_id"], []).append(state)

    profile_rmse = [
        float(
            sum(
                squared_error
                for state in states
                for squared_error in state["squared_errors"]
            )
            / sum(len(state["squared_errors"]) for state in states)
        )
        ** 0.5
        for states in profile_states_by_profile.values()
    ]
    recursive_rmse = sum(profile_rmse) / len(profile_rmse)
    next_step_rmse = (
        sum(
            float(state["next_step_error"]) ** 2
            for states in profile_states_by_profile.values()
            for state in states
        )
        / sum(len(states) for states in profile_states_by_profile.values())
    ) ** 0.5
    error_rows = [
        error_row
        for state in profile_states
        for error_row in state["errors"]
    ]
    return recursive_rmse, next_step_rmse, pd.DataFrame(error_rows)


def _make_model(params: dict[str, Any]) -> XGBRegressor:
    return XGBRegressor(**BASE_MODEL_PARAMS, **params)


def hypam_sweep(
    grid: dict[str, list[Any]] | None = None,
) -> pd.DataFrame:
    """Tune the load single-step XGBoost model using train and validation only."""
    selected_grid = LOAD_GRID_1 if grid is None else grid
    param_configs = build_param_grid(selected_grid)

    training_frame = _read_split("train")
    feature_columns = _get_feature_columns(training_frame)
    x_train, y_train = _get_valid_data(training_frame, feature_columns, "train")
    validation_frame = _read_split("validation")
    _get_valid_data(validation_frame, feature_columns, "validation")

    rows = []
    best_score = float("inf")
    best_validation_errors = None
    for index, params in enumerate(param_configs, 1):
        model = _make_model(params)
        model.fit(x_train, y_train)
        recursive_rmse, next_step_rmse, validation_errors = _score_recursive_full_day(
            model,
            validation_frame,
            feature_columns,
            "validation",
        )
        rows.append(
            {
                **params,
                "validation_recursive_rmse": recursive_rmse,
                "validation_next_step_rmse": next_step_rmse,
                "forecast_start_timesteps": ",".join(
                    map(str, FORECAST_START_TIMESTEPS)
                ),
            }
        )
        if recursive_rmse < best_score:
            best_score = recursive_rmse
            best_validation_errors = validation_errors
        print(
            f"[{index}/{len(param_configs)}] "
            f"params={params} recursive_rmse={recursive_rmse:.6f} "
            f"next_step_rmse={next_step_rmse:.6f}"
        )

    results = pd.DataFrame(rows).sort_values(
        "validation_recursive_rmse",
        ignore_index=True,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "grid_results.csv", index=False)
    with (OUTPUT_DIR / "grid.json").open("w", encoding="utf-8") as file:
        json.dump(selected_grid, file, indent=2)

    best_result = results.iloc[0]
    best_params = {
        key: best_result[key].item()
        if hasattr(best_result[key], "item")
        else best_result[key]
        for key in selected_grid
    }
    with (OUTPUT_DIR / "best_params.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "params": best_params,
                "validation_recursive_rmse": float(
                    best_result["validation_recursive_rmse"]
                ),
                "validation_next_step_rmse": float(
                    best_result["validation_next_step_rmse"]
                ),
                "forecast_start_timesteps": list(FORECAST_START_TIMESTEPS),
            },
            file,
            indent=2,
        )
    if best_validation_errors is None:
        raise RuntimeError("The hyperparameter sweep produced no validation results")
    best_validation_errors.to_csv(
        OUTPUT_DIR / "best_validation_profile_errors.csv",
        index=False,
    )

    print(f"Best params: {best_params}")
    print(
        "Best validation recursive full-day RMSE: "
        f"{best_result['validation_recursive_rmse']:.6f}"
    )
    print(f"Best validation next-step RMSE: {best_result['validation_next_step_rmse']:.6f}")
    print(f"Saved sweep results to {OUTPUT_DIR}")
    return results


def evaluate_best_on_test() -> dict[str, Any]:
    """Refit the selected model on train+validation and evaluate it on test."""
    best_params_path = OUTPUT_DIR / "best_params.json"
    if not best_params_path.is_file():
        raise FileNotFoundError(
            f"No selected hyperparameters found at {best_params_path}; run the sweep first"
        )
    with best_params_path.open(encoding="utf-8") as file:
        best_result = json.load(file)
    best_params = best_result["params"]

    training_frame = _read_split("train")
    feature_columns = _get_feature_columns(training_frame)
    x_train, y_train = _get_valid_data(training_frame, feature_columns, "train")
    validation_frame = _read_split("validation")
    x_validation, y_validation = _get_valid_data(
        validation_frame,
        feature_columns,
        "validation",
    )
    x_train_validation = pd.concat([x_train, x_validation], ignore_index=True)
    y_train_validation = pd.concat([y_train, y_validation], ignore_index=True)

    test_frame = _read_split("test")
    _, y_test = _get_valid_data(test_frame, feature_columns, "test")
    model = _make_model(best_params)
    model.fit(x_train_validation, y_train_validation)
    test_recursive_rmse, test_next_step_rmse, test_errors = _score_recursive_full_day(
        model,
        test_frame,
        feature_columns,
        "test",
    )

    result = {
        "params": best_params,
        "validation_recursive_rmse": best_result["validation_recursive_rmse"],
        "validation_next_step_rmse": best_result["validation_next_step_rmse"],
        "forecast_start_timesteps": list(FORECAST_START_TIMESTEPS),
        "test_recursive_rmse": test_recursive_rmse,
        "test_next_step_rmse": test_next_step_rmse,
        "train_validation_rows": len(y_train_validation),
        "test_rows": len(y_test),
    }
    with (OUTPUT_DIR / "test_evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    test_errors.to_csv(OUTPUT_DIR / "test_profile_errors.csv", index=False)

    print(f"Final test recursive full-day RMSE: {test_recursive_rmse:.6f}")
    print(f"Final test next-step RMSE: {test_next_step_rmse:.6f}")
    print(f"Saved final test evaluation to {OUTPUT_DIR / 'test_evaluation.json'}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune the load single-step XGBoost model."
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the selected hyperparameters once on the held-out test split.",
    )
    arguments = parser.parse_args()
    if arguments.evaluate_test:
        evaluate_best_on_test()
    else:
        hypam_sweep()
