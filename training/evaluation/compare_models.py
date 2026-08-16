from __future__ import annotations

from pathlib import Path
import json
import math
import pickle
import sys
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor


def _bootstrap_repo_root() -> Path:
    repo_root = next((p for p in Path(__file__).resolve().parents if (p / "src").exists()), None)
    if repo_root is None:
        raise RuntimeError("Could not locate repository root containing 'src'.")

    repo_root = Path(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


REPO_ROOT = _bootstrap_repo_root()

from src.config import Config  # noqa: E402
from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig  # noqa: E402
from training._features.base_load_features import get_base_load_features  # noqa: E402
from training._features.ev_status_features import get_ev_status_features  # noqa: E402
from training._features.pv_gen_features import get_pv_gen_features  # noqa: E402
from training.model_artifacts import render_training_params_manifest  # noqa: E402


TARGETS = ("base_load", "pv_gen", "ev1_status", "ev2_status")
FAMILY_TO_FEATURE_FAMILY = {
    "xgb": "xgboost",
    "rf": "random_forest",
    "ridge": "ridge",
}


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "none", "nan"}


def _parse_manifest_scalar(value: str) -> Any:
    text = value.strip()
    lowered = text.lower()

    if lowered == "none":
        return None

    try:
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _load_params_from_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing training params manifest: {manifest_path}")

    params: dict[str, Any] = {}
    in_params_block = False

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not in_params_block:
            if line.strip() == "params:":
                in_params_block = True
            continue

        if not line.strip():
            continue

        if not line.startswith("  "):
            break

        key, _, raw_value = line.strip().partition(":")
        if not _:
            raise ValueError(f"Invalid manifest line in {manifest_path}: {line!r}")
        params[key.strip()] = _parse_manifest_scalar(raw_value)

    if not params:
        raise ValueError(f"No params block found in training params manifest: {manifest_path}")

    return params


def _parse_xgb_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "learning_rate": float(params["learning_rate"]),
        "n_estimators": int(params["n_estimators"]),
        "max_depth": int(params["max_depth"]),
    }


def _parse_rf_params(params: dict[str, Any]) -> dict[str, Any]:
    max_depth_value = params.get("max_depth")
    if _is_missing(max_depth_value) or str(max_depth_value).lower() == "none":
        parsed_max_depth = None
    else:
        parsed_max_depth = int(float(max_depth_value))

    max_features_value = params.get("max_features")
    if _is_missing(max_features_value):
        parsed_max_features = None
    else:
        max_features_text = str(max_features_value).strip().lower()
        if max_features_text in {"sqrt", "log2", "none"}:
            parsed_max_features = None if max_features_text == "none" else max_features_text
        else:
            parsed_max_features = float(max_features_value)

    return {
        "n_estimators": int(float(params["n_estimators"])),
        "max_depth": parsed_max_depth,
        "min_samples_leaf": int(float(params["min_samples_leaf"])),
        "max_features": parsed_max_features,
        "random_state": 42,
        "n_jobs": -1,
    }


def _parse_ridge_params(params: dict[str, Any]) -> dict[str, Any]:
    return {"alpha": float(params["alpha"])}


def _model_class(family: str, target: str):
    if family == "xgb" and target in {"ev1_status", "ev2_status"}:
        return XGBClassifier
    if family == "xgb":
        return XGBRegressor
    return None


def _load_model(family: str, target: str, model_path: Path):
    model_cls = _model_class(family, target)
    if model_cls is not None:
        model = model_cls()
        model.load_model(str(model_path))
        return model

    with open(model_path, "rb") as fh:
        return pickle.load(fh)


def _feature_family_for_model(family: str) -> str:
    return FAMILY_TO_FEATURE_FAMILY[family]


def _feature_columns_for_target(family: str, target: str) -> list[str]:
    family_features = ModelConfig.MODEL_FEATURES_BY_FAMILY[_feature_family_for_model(family)]
    if target in {"ev1_status", "ev2_status"}:
        return family_features["ev_status"]
    return family_features[target]


def _target_spec(target: str) -> tuple[list[int], Any, str, str]:
    if target == "base_load":
        return list(Config.ALL_PLAYER_IDS), get_base_load_features, "next_value", "regression"
    if target == "pv_gen":
        return list(Config.PLAYERS_WITH_PV), get_pv_gen_features, "next_value", "regression"
    if target == "ev1_status":
        return list(Config.ALL_PLAYER_IDS), get_ev_status_features, "next_state", "classification"
    if target == "ev2_status":
        return list(Config.ALL_PLAYER_IDS), get_ev_status_features, "next_state", "classification"
    raise ValueError(f"Unknown target '{target}'.")


def _apply_target_filter(feature_df: pd.DataFrame, target: str) -> pd.DataFrame:
    filtered_df = feature_df.copy()

    if target == "pv_gen":
        window = Config.PV_GENERATION_WINDOW_ALLOWED
        return filtered_df.loc[
            filtered_df["timestep"].between(
                int(window["earliest_start"]),
                int(window["latest_end"]),
            )
        ].copy()

    if target in {"ev1_status", "ev2_status"}:
        ev_key = target.split("_")[0]

        keep_mask = pd.Series(False, index=filtered_df.index)
        for commute_window in Config.EV_COMMUTE_WINDOWS_ALLOWED[ev_key]:
            keep_mask |= filtered_df["timestep"].between(
                int(commute_window["earliest_start"]),
                int(commute_window["latest_end"]),
            )

        return filtered_df.loc[keep_mask].copy()

    return filtered_df


def _metrics(y_true: pd.Series, y_pred: pd.Series, target_kind: str) -> dict[str, float]:
    if target_kind == "regression":
        mse = mean_squared_error(y_true, y_pred)
        return {
            "rmse": float(mse ** 0.5),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def _normalise_training_params(family: str, raw_params: dict[str, Any]) -> dict[str, Any]:
    if family == "xgb":
        return _parse_xgb_params(raw_params)
    if family == "rf":
        return _parse_rf_params(raw_params)
    return _parse_ridge_params(raw_params)


def _training_manifest_path(family: str, target: str) -> Path:
    model_dir = Config.MODEL_FAMILY_CONFIGS[family].target_model_dirs[target]
    return Path(model_dir) / "training_params.txt"


def _model_path_for(family: str, target: str, fold_id: str) -> Path:
    model_dir = Config.MODEL_FAMILY_CONFIGS[family].target_model_dirs[target]
    suffix = Config.MODEL_FAMILY_CONFIGS[family].file_suffix
    return Path(model_dir / f"{fold_id}{suffix}")


def _manifest_text(family: str, target: str, params: dict[str, Any]) -> str:
    return render_training_params_manifest(
        family=family,
        target=target,
        fold_ids=Config.FOLD_IDS,
        params=params,
    )


def evaluate_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for family in ("xgb", "rf", "ridge"):
        for target in TARGETS:
            household_ids, feature_builder, label_column, target_kind = _target_spec(target)
            manifest_path = _training_manifest_path(family, target)
            params = _normalise_training_params(family, _load_params_from_manifest(manifest_path))

            feature_df = feature_builder(household_ids)
            feature_df = _apply_target_filter(feature_df, target)
            feature_columns = _feature_columns_for_target(family, target)

            if feature_df.empty:
                continue

            if target in {"ev1_status", "ev2_status"}:
                feature_df = feature_df.loc[feature_df["ev_key"] == target.split("_")[0]].copy()

            feature_df["household_id"] = feature_df["household_id"].astype(int)
            feature_df["fold_id"] = feature_df["household_id"].map(
                lambda household_id: Config.get_fold_for_player(target, int(household_id))
            )

            y_true_parts: list[pd.Series] = []
            y_pred_parts: list[pd.Series] = []

            for fold_id in Config.FOLD_IDS:
                fold_df = feature_df.loc[feature_df["fold_id"] == fold_id].copy()
                if fold_df.empty:
                    continue

                model_path = _model_path_for(family, target, fold_id)
                model = _load_model(family, target, model_path)
                predictions = model.predict(fold_df[feature_columns].to_numpy())

                y_true_parts.append(fold_df[label_column].reset_index(drop=True))
                y_pred_parts.append(pd.Series(predictions).reset_index(drop=True))

                for row, pred in zip(fold_df.itertuples(index=False), predictions):
                    detail_rows.append(
                        {
                            "family": family,
                            "target": target,
                            "fold_id": fold_id,
                            "household_id": int(row.household_id),
                            "timestep": int(row.timestep),
                            "ev_key": getattr(row, "ev_key", ""),
                            "y_true": getattr(row, label_column),
                            "y_pred": pred,
                        }
                    )

            if not y_true_parts:
                continue

            y_true = pd.concat(y_true_parts, ignore_index=True)
            y_pred = pd.concat(y_pred_parts, ignore_index=True)
            metric_values = _metrics(y_true, y_pred, target_kind)

            summary_rows.append(
                {
                    "family": family,
                    "target": target,
                    "n_rows": int(len(feature_df)),
                    "n_households": int(feature_df["household_id"].nunique()),
                    "n_timesteps": int(feature_df["timestep"].nunique()),
                    "primary_metric_name": "rmse" if target_kind == "regression" else "accuracy",
                    "primary_metric_value": metric_values["rmse"] if target_kind == "regression" else metric_values["accuracy"],
                    "secondary_metric_name": "mae" if target_kind == "regression" else "balanced_accuracy",
                    "secondary_metric_value": metric_values["mae"] if target_kind == "regression" else metric_values["balanced_accuracy"],
                    "training_params": json.dumps(params, sort_keys=True, default=str),
                    "training_params_file": str(manifest_path),
                    "training_params_preview": _manifest_text(family, target, params),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(["target", "family"]).reset_index(drop=True)
    details_df = pd.DataFrame(detail_rows).sort_values(["target", "family", "household_id", "timestep"]).reset_index(drop=True)
    return summary_df, details_df


def main() -> None:
    output_dir = REPO_ROOT / "reports" / "model_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df, details_df = evaluate_models()
    summary_path = output_dir / "model_comparison_summary.csv"
    details_path = output_dir / "model_comparison_details.csv"

    while summary_path.exists() or details_path.exists():
        counter = 1
        while summary_path.exists():
            summary_path = output_dir / f"model_comparison_summary_{counter}.csv"
            counter += 1
        counter = 1
        while details_path.exists():
            details_path = output_dir / f"model_comparison_details_{counter}.csv"
            counter += 1

    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)

    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved details to {details_path}")


if __name__ == "__main__":
    main()