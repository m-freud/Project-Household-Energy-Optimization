import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier, XGBRegressor

from src.runtime_config import RuntimeConfig
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._ev_status import (
    _build_ev_status_features,
    _try_bypass as _ev_try_bypass,
)
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._pv_gen import (
    _build_pv_gen_features,
    _try_bypass as _pv_try_bypass,
)
from training.tuning.pred_hypam_sweep import (
    DIRECT_HORIZONS,
    FORECAST_START_TIMESTEPS,
    LOAD_GRID_1,
    build_param_grid,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
FEATURE_ROOT = ROOT_DIR / "training" / "features" / "feature_parquets"
OUTPUT_ROOT = Path(__file__).resolve().parent
PV_FILENAME_PATTERN = "de_pv_features_*.parquet"
EV_FILENAME_PATTERNS = {
    "ev1": "ev1_status_features_*.parquet",
    "ev2": "ev2_status_features_*.parquet",
}
IDENTIFIER_COLUMNS = {"profile_id", "household_id", "date", "ev_key", "phase"}
TARGET_PATTERN = re.compile(r"y\d+")
EV_CLASSES = (0, 1, 2)
BASE_MODEL_PARAMS: dict[str, Any] = {
    "random_state": 42,
    "n_jobs": -1,
}


def _validate_target(target: str, horizon: int | None = None) -> None:
    if target not in {"pv", "ev1", "ev2"}:
        raise ValueError(f"Unsupported target '{target}'; expected pv, ev1, or ev2")
    if horizon is not None and (target != "pv" or horizon not in DIRECT_HORIZONS):
        raise ValueError(
            "Direct horizons are supported only for PV and must be one of "
            f"{DIRECT_HORIZONS}"
        )


def _target_column(target: str, horizon: int | None = None) -> str:
    if target == "pv":
        return "next_value" if horizon is None else f"y{horizon}"
    if horizon is not None:
        raise ValueError("EV sweeps support single-step prediction only")
    return "next_state"


def _read_split(target: str, split: str) -> pd.DataFrame:
    _validate_target(target)
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported data split '{split}'")
    pattern = (
        PV_FILENAME_PATTERN if target == "pv" else EV_FILENAME_PATTERNS[target]
    )
    split_dir = FEATURE_ROOT / target / split
    parquet_paths = sorted(split_dir.glob(pattern))
    if not parquet_paths:
        raise FileNotFoundError(f"No {target} feature Parquets found in {split_dir}")

    frame = pd.concat(
        [pd.read_parquet(path) for path in parquet_paths],
        ignore_index=True,
    )
    target_column = _target_column(target)
    if target_column not in frame.columns:
        raise ValueError(f"{split} data is missing target column '{target_column}'")
    return frame


def _get_feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    target_columns = {
        column
        for column in frame.columns
        if TARGET_PATTERN.fullmatch(column)
    }
    excluded_columns = IDENTIFIER_COLUMNS | target_columns | {
        _target_column(target)
    }
    numeric_columns = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in excluded_columns
    ]
    if not numeric_columns:
        raise ValueError(f"No numeric model features found for {target}")
    return numeric_columns


def _get_valid_data(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    target: str,
    split: str,
    *,
    forecast_starts_only: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    missing_columns = [
        column
        for column in [*feature_columns, target_column, "timestep"]
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"{split} data is missing columns: {missing_columns}")

    valid_rows = frame[target_column].ne(-1)
    if forecast_starts_only:
        valid_rows &= frame["timestep"].isin(FORECAST_START_TIMESTEPS)
    valid_frame = frame.loc[valid_rows]
    if valid_frame.empty:
        raise ValueError(f"No valid '{target_column}' rows found in {split} data")
    checked_columns = [*feature_columns, target_column]
    if valid_frame[checked_columns].isna().any().any():
        raise ValueError(f"{target} {split} data contains missing features or targets")
    non_numeric_columns = [
        column
        for column in checked_columns
        if not pd.api.types.is_numeric_dtype(valid_frame[column])
    ]
    if non_numeric_columns:
        raise ValueError(
            f"{target} {split} data columns must be numeric: {non_numeric_columns}"
        )
    if target != "pv":
        labels = set(valid_frame[target_column].unique().tolist())
        if not labels.issubset(EV_CLASSES):
            raise ValueError(f"Unexpected EV labels in {split}: {sorted(labels)}")
    return valid_frame[feature_columns], valid_frame[target_column]


def _make_model(target: str, params: dict[str, Any]) -> XGBRegressor | XGBClassifier:
    if target == "pv":
        return XGBRegressor(
            **BASE_MODEL_PARAMS,
            objective="reg:squarederror",
            **params,
        )
    return XGBClassifier(
        **BASE_MODEL_PARAMS,
        objective="multi:softprob",
        num_class=len(EV_CLASSES),
        eval_metric="mlogloss",
        **params,
    )


def _profile_rmse(frame: pd.DataFrame) -> float:
    profile_scores = frame.groupby("profile_id")["error"].apply(
        lambda errors: float(np.sqrt(np.mean(np.square(errors))))
    )
    return float(profile_scores.mean())


def _classification_metrics(actual: list[int], predicted: list[int]) -> dict[str, float]:
    present_labels = set(actual)
    recalls = [
        sum(
            target == label and estimate == label
            for target, estimate in zip(actual, predicted)
        )
        / sum(target == label for target in actual)
        for label in EV_CLASSES
        if label in present_labels
    ]
    return {
        "macro_f1": float(
            f1_score(actual, predicted, labels=EV_CLASSES, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
    }


def _runtime_features(
    target: str,
    state: dict[str, Any],
    timestep: int,
    feature_columns: list[str],
) -> dict[str, float | int]:
    if target == "pv":
        daylight_window = RuntimeConfig.PV_GENERATION_WINDOW_ALLOWED
        return _build_pv_gen_features(
            current_timestep=timestep,
            current_pv_gen=float(state["current"]),
            pv_history=state["history"],
            daylight_start=int(daylight_window["earliest_start"]),
            daylight_end=int(daylight_window["latest_end"]),
            required_features=feature_columns,
        )

    windows = RuntimeConfig.EV_COMMUTE_WINDOWS_ALLOWED[target]
    return _build_ev_status_features(
        {
            "timestep": timestep,
            "status": int(state["current"]),
            "status_history": state["history"],
            "start1_earliest": int(windows[0]["earliest_start"]),
            "end1_latest": int(windows[0]["latest_end"]),
            "start2_earliest": int(windows[1]["earliest_start"]),
            "end2_latest": int(windows[1]["latest_end"]),
            "max_commute_steps_1": int(windows[0]["max_unavailable_steps"]),
            "max_commute_steps_2": int(windows[1]["max_unavailable_steps"]),
        },
        required_features=feature_columns,
    )


def _runtime_bypass(target: str, state: dict[str, Any], timestep: int) -> int | float | None:
    if target == "pv":
        return _pv_try_bypass(timestep)
    windows = RuntimeConfig.EV_COMMUTE_WINDOWS_ALLOWED[target]
    return _ev_try_bypass(
        current_time=timestep,
        ev_key=target,
        status_history=state["history"],
        current_status=int(state["current"]),
    )


def _build_runtime_feature_frame(
    target: str,
    frame: pd.DataFrame,
    feature_columns: list[str],
    split: str,
) -> pd.DataFrame:
    value_column = "pv_gen" if target == "pv" else "status"
    required_columns = {"profile_id", "timestep", value_column}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{split} data is missing columns: {missing_columns}")

    runtime_frame = frame.copy()
    for profile_id, profile in frame.groupby("profile_id", sort=False):
        profile = profile.sort_values("timestep")
        if profile["timestep"].tolist() != list(range(1, 97)):
            raise ValueError(
                f"Expected profile {profile_id} in {split} data to contain "
                "exactly timesteps 1-96"
            )
        actual = profile[value_column].to_numpy()
        if pd.isna(actual).any():
            raise ValueError(f"Profile {profile_id} in {split} data has missing values")

        feature_rows = []
        for index, value in enumerate(actual):
            state = {
                "current": value.item(),
                "history": actual[:index].astype(int).tolist()
                if target != "pv"
                else actual[:index].astype(float).tolist(),
            }
            built_features = _runtime_features(
                target,
                state,
                index + 1,
                feature_columns,
            )
            missing_features = set(feature_columns) - set(built_features)
            if missing_features:
                raise ValueError(
                    f"Runtime {target} feature builder omitted "
                    f"{sorted(missing_features)}"
                )
            feature_rows.append(
                {column: built_features[column] for column in feature_columns}
            )
        runtime_features = pd.DataFrame(feature_rows, index=profile.index)
        runtime_frame.loc[profile.index, feature_columns] = runtime_features

    if runtime_frame[feature_columns].isna().any().any():
        raise ValueError(f"Generated {target} {split} features contain missing values")
    return runtime_frame


def _score_recursive_single_step(
    model: XGBRegressor | XGBClassifier,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    split: str,
) -> tuple[dict[str, float], dict[str, float], pd.DataFrame]:
    value_column = "pv_gen" if target == "pv" else "status"
    required_columns = {
        "profile_id",
        "household_id",
        "timestep",
        value_column,
    }
    if target == "pv":
        required_columns.add("date")
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{split} data is missing profile columns: {missing_columns}")

    states: list[dict[str, Any]] = []
    for profile_id, profile in frame.groupby("profile_id", sort=False):
        profile = profile.sort_values("timestep")
        if profile["timestep"].tolist() != list(range(1, 97)):
            raise ValueError(
                f"Expected profile {profile_id} in {split} data to contain "
                "exactly timesteps 1-96"
            )
        actual = profile[value_column].to_numpy()
        if pd.isna(actual).any():
            raise ValueError(f"Profile {profile_id} in {split} data has missing values")
        if target != "pv" and not set(actual).issubset(EV_CLASSES):
            raise ValueError(f"Profile {profile_id} in {split} data has invalid EV labels")
        static_context: dict[str, Any] = {}
        if target == "pv":
            static_context["date"] = profile["date"].iloc[0]
        for start_timestep in FORECAST_START_TIMESTEPS:
            states.append(
                {
                    "profile_id": profile_id,
                    "household_id": profile["household_id"].iloc[0],
                    **static_context,
                    "actual": actual,
                    "start_timestep": start_timestep,
                    "history": actual[: start_timestep - 1].astype(int).tolist()
                    if target != "pv"
                    else actual[: start_timestep - 1].astype(float).tolist(),
                    "current": actual[start_timestep - 1].item(),
                    "expected_start_features": profile.iloc[
                        start_timestep - 1
                    ][feature_columns].to_dict(),
                    "errors": [],
                    "next_step_actual": None,
                    "next_step_predicted": None,
                }
            )
    if not states:
        raise ValueError(f"No profile-days found in {split} data")

    for timestep in range(1, 96):
        active_states = [
            state
            for state in states
            if state["start_timestep"] <= timestep
            and len(state["errors"]) < 96 - state["start_timestep"]
        ]
        if not active_states:
            continue

        model_states = []
        feature_rows = []
        for state in active_states:
            bypass = _runtime_bypass(target, state, timestep)
            if bypass is not None:
                if target != "pv":
                    state["history"].append(int(state["current"]))
                prediction: int | float
                prediction = int(bypass) if target != "pv" else float(bypass)
                _record_recursive_prediction(state, timestep, prediction, target)
                state["current"] = prediction
                continue

            runtime_features = _runtime_features(
                target,
                state,
                timestep,
                feature_columns,
            )
            if timestep == state["start_timestep"]:
                for column in feature_columns:
                    if not math.isclose(
                        float(runtime_features[column]),
                        float(state["expected_start_features"][column]),
                        rel_tol=0.0,
                        abs_tol=0.00051,
                    ):
                        raise ValueError(
                            f"Rebuilt {target} feature mismatch in {split} for "
                            f"profile {state['profile_id']} at timestep {timestep}: "
                            f"{column}"
                        )
            state["history"].append(
                int(state["current"])
                if target != "pv"
                else float(state["current"])
            )
            model_states.append(state)
            feature_rows.append(runtime_features)

        if not model_states:
            continue
        model_input = pd.DataFrame(feature_rows, columns=feature_columns)
        if model_input.isna().any().any():
            raise ValueError(f"Generated recursive {target} features contain missing values")
        predictions = model.predict(model_input)
        for state, prediction in zip(model_states, predictions):
            if target == "pv":
                state["history"].append(float(state["current"]))
                predicted_value: int | float = float(prediction)
            else:
                predicted_class = int(prediction)
                if predicted_class not in EV_CLASSES or predicted_class != prediction:
                    raise ValueError(f"EV model predicted invalid status class {prediction}")
                predicted_value = predicted_class
            _record_recursive_prediction(state, timestep, predicted_value, target)
            state["current"] = predicted_value

    errors = pd.DataFrame(
        error for state in states for error in state["errors"]
    )
    next_step_rows = [
        state
        for state in states
        if state["next_step_actual"] is not None
    ]

    if target == "pv":
        recursive_metrics = {
            "recursive_rmse": _profile_rmse(errors),
            "pooled_rmse": float(np.sqrt(np.mean(np.square(errors["error"])))),
        }
        next_step_errors = np.asarray(
            [
                float(state["next_step_predicted"])
                - float(state["next_step_actual"])
                for state in next_step_rows
            ],
            dtype=float,
        )
        next_step_metrics = {
            "next_step_rmse": float(np.sqrt(np.mean(np.square(next_step_errors))))
        }
        return recursive_metrics, next_step_metrics, errors

    actual_labels = errors["actual"].astype(int).tolist()
    predicted_labels = errors["predicted"].astype(int).tolist()
    recursive_metrics = _classification_metrics(actual_labels, predicted_labels)
    next_step_metrics = _classification_metrics(
        [int(state["next_step_actual"]) for state in next_step_rows],
        [int(state["next_step_predicted"]) for state in next_step_rows],
    )
    return recursive_metrics, {
        f"next_step_{name}": value for name, value in next_step_metrics.items()
    }, errors


def _record_recursive_prediction(
    state: dict[str, Any],
    timestep: int,
    prediction: int | float,
    target: str,
) -> None:
    actual = state["actual"][timestep]
    row: dict[str, Any] = {
        "profile_id": state["profile_id"],
        "household_id": state["household_id"],
        "start_timestep": state["start_timestep"],
        "timestep": timestep + 1,
        "actual": actual,
        "predicted": prediction,
    }
    if target == "pv":
        row["date"] = state["date"]
        row["error"] = float(prediction) - float(actual)
    else:
        row["correct"] = int(prediction == actual)
    state["errors"].append(row)
    if timestep == state["start_timestep"]:
        state["next_step_actual"] = actual
        state["next_step_predicted"] = prediction


def _score_direct_pv(
    model: XGBRegressor,
    frame: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    split: str,
) -> tuple[dict[str, float], tuple[int, ...], pd.DataFrame]:
    target_column = f"y{horizon}"
    missing_columns = [
        column
        for column in [
            "profile_id",
            "household_id",
            "date",
            "timestep",
            *feature_columns,
            target_column,
        ]
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"{split} data is missing columns: {missing_columns}")

    evaluation_rows = frame.loc[
        frame["timestep"].isin(FORECAST_START_TIMESTEPS)
        & frame[target_column].ne(-1)
    ].copy()
    if evaluation_rows.empty:
        raise ValueError(
            f"No valid '{target_column}' rows at forecast starts in {split} data"
        )
    if evaluation_rows[[*feature_columns, target_column]].isna().any().any():
        raise ValueError(f"PV {split} data contains missing features or targets")
    predictions = model.predict(evaluation_rows[feature_columns])
    evaluation_rows["predicted"] = predictions
    evaluation_rows["actual"] = evaluation_rows[target_column]
    evaluation_rows["error"] = evaluation_rows["predicted"] - evaluation_rows["actual"]
    evaluation_rows["start_timestep"] = evaluation_rows["timestep"]
    evaluation_rows["target_timestep"] = evaluation_rows["timestep"] + horizon
    daily_rmse = _profile_rmse(evaluation_rows)
    metrics = {
        "direct_rmse": daily_rmse,
        "pooled_rmse": float(np.sqrt(np.mean(np.square(evaluation_rows["error"])))),
    }
    starts = tuple(sorted(int(value) for value in evaluation_rows["timestep"].unique()))
    columns = [
        "profile_id",
        "household_id",
        "date",
        "start_timestep",
        "target_timestep",
        "actual",
        "predicted",
        "error",
    ]
    return metrics, starts, evaluation_rows[columns].reset_index(drop=True)


def _sweep_single_step(target: str) -> pd.DataFrame:
    training = _read_split(target, "train")
    feature_columns = _get_feature_columns(training, target)
    training = _build_runtime_feature_frame(
        target,
        training,
        feature_columns,
        "train",
    )
    target_column = _target_column(target)
    x_train, y_train = _get_valid_data(
        training,
        feature_columns,
        target_column,
        target,
        "train",
    )
    validation = _build_runtime_feature_frame(
        target,
        _read_split(target, "validation"),
        feature_columns,
        "validation",
    )
    _get_valid_data(
        validation,
        feature_columns,
        target_column,
        target,
        "validation",
    )

    rows = []
    best_score = -float("inf") if target != "pv" else float("inf")
    best_errors = None
    for index, params in enumerate(build_param_grid(LOAD_GRID_1), 1):
        model = _make_model(target, params)
        model.fit(x_train, y_train)
        recursive_metrics, next_metrics, errors = _score_recursive_single_step(
            model,
            validation,
            feature_columns,
            target,
            "validation",
        )
        score_column = "recursive_rmse" if target == "pv" else "macro_f1"
        score = recursive_metrics[score_column]
        row = {
            **params,
            **{f"validation_{key}": value for key, value in recursive_metrics.items()},
            **{f"validation_{key}": value for key, value in next_metrics.items()},
            "forecast_start_timesteps": ",".join(
                map(str, FORECAST_START_TIMESTEPS)
            ),
        }
        rows.append(row)
        is_better = score < best_score if target == "pv" else score > best_score
        if is_better:
            best_score = score
            best_errors = errors
        print(
            f"[{index}/60] target={target} params={params} "
            f"validation_{score_column}={score:.6f}"
        )

    score_column = "validation_recursive_rmse" if target == "pv" else "validation_macro_f1"
    results = pd.DataFrame(rows).sort_values(
        score_column,
        ascending=target == "pv",
        ignore_index=True,
    )
    output_dir = OUTPUT_ROOT / target / "single_step"
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "grid_results.csv", index=False)
    with (output_dir / "grid.json").open("w", encoding="utf-8") as file:
        json.dump(LOAD_GRID_1, file, indent=2)
    best = results.iloc[0]
    best_params = {
        key: best[key].item() if hasattr(best[key], "item") else best[key]
        for key in LOAD_GRID_1
    }
    if best_errors is None:
        raise RuntimeError(f"No validation results were produced for {target}")
    with (output_dir / "best_params.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "params": best_params,
                "validation_metrics": {
                    column.removeprefix("validation_"): float(best[column])
                    for column in best.index
                    if column.startswith("validation_")
                },
                "forecast_start_timesteps": list(FORECAST_START_TIMESTEPS),
            },
            file,
            indent=2,
        )
    best_errors.to_csv(output_dir / "best_validation_profile_errors.csv", index=False)
    print(f"Best validation results saved to {output_dir}")
    return results


def _evaluate_single_step_test(target: str) -> dict[str, Any]:
    output_dir = OUTPUT_ROOT / target / "single_step"
    best_path = output_dir / "best_params.json"
    if not best_path.is_file():
        raise FileNotFoundError(f"Missing selected hyperparameters at {best_path}")
    with best_path.open(encoding="utf-8") as file:
        best = json.load(file)
    params = best["params"]
    training = _read_split(target, "train")
    feature_columns = _get_feature_columns(training, target)
    training = _build_runtime_feature_frame(
        target,
        training,
        feature_columns,
        "train",
    )
    validation = _build_runtime_feature_frame(
        target,
        _read_split(target, "validation"),
        feature_columns,
        "validation",
    )
    target_column = _target_column(target)
    x_train, y_train = _get_valid_data(
        training,
        feature_columns,
        target_column,
        target,
        "train",
    )
    x_validation, y_validation = _get_valid_data(
        validation,
        feature_columns,
        target_column,
        target,
        "validation",
    )
    test = _build_runtime_feature_frame(
        target,
        _read_split(target, "test"),
        feature_columns,
        "test",
    )
    _, y_test = _get_valid_data(
        test,
        feature_columns,
        target_column,
        target,
        "test",
    )
    model = _make_model(target, params)
    model.fit(
        pd.concat([x_train, x_validation], ignore_index=True),
        pd.concat([y_train, y_validation], ignore_index=True),
    )
    recursive_metrics, next_metrics, errors = _score_recursive_single_step(
        model,
        test,
        feature_columns,
        target,
        "test",
    )
    result = {
        "params": params,
        "validation_metrics": best["validation_metrics"],
        "test_metrics": recursive_metrics,
        "test_next_step_metrics": next_metrics,
        "forecast_start_timesteps": list(FORECAST_START_TIMESTEPS),
        "train_validation_rows": len(y_train) + len(y_validation),
        "test_rows": len(y_test),
    }
    with (output_dir / "test_evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    errors.to_csv(output_dir / "test_profile_errors.csv", index=False)
    print(f"Final {target} test results saved to {output_dir / 'test_evaluation.json'}")
    return result


def _sweep_direct_pv(horizon: int) -> pd.DataFrame:
    _validate_target("pv", horizon)
    target_column = f"y{horizon}"
    training = _read_split("pv", "train")
    feature_columns = _get_feature_columns(training, "pv")
    training = _build_runtime_feature_frame(
        "pv",
        training,
        feature_columns,
        "train",
    )
    x_train, y_train = _get_valid_data(
        training,
        feature_columns,
        target_column,
        "pv",
        "train",
    )
    validation = _build_runtime_feature_frame(
        "pv",
        _read_split("pv", "validation"),
        feature_columns,
        "validation",
    )
    _get_valid_data(
        validation,
        feature_columns,
        target_column,
        "pv",
        "validation",
        forecast_starts_only=True,
    )

    rows = []
    best_errors = None
    for index, params in enumerate(build_param_grid(LOAD_GRID_1), 1):
        model = _make_model("pv", params)
        model.fit(x_train, y_train)
        metrics, starts, errors = _score_direct_pv(
            model,
            validation,
            feature_columns,
            horizon,
            "validation",
        )
        rows.append(
            {
                **params,
                "validation_direct_rmse": metrics["direct_rmse"],
                "validation_pooled_rmse": metrics["pooled_rmse"],
                "evaluation_start_timesteps": ",".join(map(str, starts)),
            }
        )
        if best_errors is None or metrics["direct_rmse"] < best_errors[0]:
            best_errors = (metrics["direct_rmse"], errors)
        print(
            f"[{index}/60] target=pv horizon={horizon} params={params} "
            f"direct_rmse={metrics['direct_rmse']:.6f}"
        )

    results = pd.DataFrame(rows).sort_values(
        "validation_direct_rmse",
        ignore_index=True,
    )
    output_dir = OUTPUT_ROOT / "pv" / f"step_{horizon}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "grid_results.csv", index=False)
    with (output_dir / "grid.json").open("w", encoding="utf-8") as file:
        json.dump(LOAD_GRID_1, file, indent=2)
    best = results.iloc[0]
    best_params = {
        key: best[key].item() if hasattr(best[key], "item") else best[key]
        for key in LOAD_GRID_1
    }
    if best_errors is None:
        raise RuntimeError(f"No validation results were produced for PV t+{horizon}")
    with (output_dir / "best_params.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "params": best_params,
                "horizon": horizon,
                "validation_direct_rmse": float(best["validation_direct_rmse"]),
                "validation_pooled_rmse": float(best["validation_pooled_rmse"]),
                "evaluation_start_timesteps": [
                    int(value)
                    for value in best["evaluation_start_timesteps"].split(",")
                ],
            },
            file,
            indent=2,
        )
    best_errors[1].to_csv(
        output_dir / "best_validation_profile_errors.csv",
        index=False,
    )
    print(f"Best PV t+{horizon} validation results saved to {output_dir}")
    return results


def _evaluate_direct_pv_test(horizon: int) -> dict[str, Any]:
    _validate_target("pv", horizon)
    output_dir = OUTPUT_ROOT / "pv" / f"step_{horizon}"
    best_path = output_dir / "best_params.json"
    if not best_path.is_file():
        raise FileNotFoundError(f"Missing selected hyperparameters at {best_path}")
    with best_path.open(encoding="utf-8") as file:
        best = json.load(file)
    if best.get("horizon") != horizon:
        raise ValueError(f"Saved PV parameters are not for horizon {horizon}")
    params = best["params"]
    target_column = f"y{horizon}"
    training = _read_split("pv", "train")
    feature_columns = _get_feature_columns(training, "pv")
    training = _build_runtime_feature_frame(
        "pv",
        training,
        feature_columns,
        "train",
    )
    validation = _build_runtime_feature_frame(
        "pv",
        _read_split("pv", "validation"),
        feature_columns,
        "validation",
    )
    x_train, y_train = _get_valid_data(
        training,
        feature_columns,
        target_column,
        "pv",
        "train",
    )
    x_validation, y_validation = _get_valid_data(
        validation,
        feature_columns,
        target_column,
        "pv",
        "validation",
    )
    test = _build_runtime_feature_frame(
        "pv",
        _read_split("pv", "test"),
        feature_columns,
        "test",
    )
    _get_valid_data(
        test,
        feature_columns,
        target_column,
        "pv",
        "test",
        forecast_starts_only=True,
    )
    model = _make_model("pv", params)
    model.fit(
        pd.concat([x_train, x_validation], ignore_index=True),
        pd.concat([y_train, y_validation], ignore_index=True),
    )
    metrics, starts, errors = _score_direct_pv(
        model,
        test,
        feature_columns,
        horizon,
        "test",
    )
    result = {
        "horizon": horizon,
        "target_column": target_column,
        "params": params,
        "validation_direct_rmse": best["validation_direct_rmse"],
        "validation_pooled_rmse": best["validation_pooled_rmse"],
        "test_direct_rmse": metrics["direct_rmse"],
        "test_pooled_rmse": metrics["pooled_rmse"],
        "evaluation_start_timesteps": list(starts),
        "train_validation_rows": len(y_train) + len(y_validation),
        "test_rows": len(errors),
    }
    with (output_dir / "test_evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    errors.to_csv(output_dir / "test_profile_errors.csv", index=False)
    print(f"Final PV t+{horizon} results saved to {output_dir / 'test_evaluation.json'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune recursive and direct XGBoost forecasts for PV and EV."
    )
    parser.add_argument("--target", required=True, choices=("pv", "ev1", "ev2"))
    parser.add_argument(
        "--horizon",
        type=int,
        choices=DIRECT_HORIZONS,
        help="Direct PV horizon. EV supports recursive single-step sweeps only.",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Refit selected parameters on train+validation and evaluate test once.",
    )
    args = parser.parse_args()
    _validate_target(args.target, args.horizon)
    if args.target != "pv" and args.horizon is not None:
        parser.error("EV targets support single-step sweeps only; omit --horizon")

    if args.horizon is None:
        if args.evaluate_test:
            _evaluate_single_step_test(args.target)
        else:
            _sweep_single_step(args.target)
    elif args.evaluate_test:
        _evaluate_direct_pv_test(args.horizon)
    else:
        _sweep_direct_pv(args.horizon)


if __name__ == "__main__":
    main()
