from pathlib import Path
import pickle
import sys

import pandas as pd
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# paste this to enable src. imports
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.config import Config
from training._features.base_load_features import get_base_load_features
from training._features.pv_gen_features import get_pv_gen_features
from training._features.ev_status_features import get_ev_status_features

FOLD_CSV_PATH = Path(Config.ROOT_DIR / "training" / "split" / "test_folds.csv")
TUNING_RESULTS_CSV_PATH = Path(repo_root) / "training" / "ridge_regression" / "tuning" / "results.csv"

TARGETS = ["base_load", "pv_gen", "ev1_status", "ev2_status"]

DEFAULT_RIDGE_PARAMS = {
    "base_load": {"alpha": 1.0},
    "pv_gen": {"alpha": 1.0},
    "ev1_status": {"alpha": 1.0},
    "ev2_status": {"alpha": 1.0},
}


def _load_best_params_from_tuning(default_params: dict[str, dict]) -> dict[str, dict]:
    if not TUNING_RESULTS_CSV_PATH.exists():
        print(f"Warning: tuning results not found at {TUNING_RESULTS_CSV_PATH}. Using default Ridge params.")
        return default_params

    results = pd.read_csv(TUNING_RESULTS_CSV_PATH)
    required_cols = {"target", "mean_score", "alpha"}
    missing = required_cols.difference(results.columns)
    if missing:
        print(
            f"Warning: tuning results missing columns {sorted(missing)}. Using default Ridge params."
        )
        return default_params

    best_params: dict[str, dict] = {}
    for target in TARGETS:
        target_rows = results.loc[results["target"] == target]
        if target_rows.empty:
            best_params[target] = default_params[target]
            continue

        best_row = target_rows.loc[target_rows["mean_score"].idxmin()]
        best_params[target] = {
            "alpha": float(best_row["alpha"]),
        }

    return best_params


RIDGE_PARAMS = _load_best_params_from_tuning(DEFAULT_RIDGE_PARAMS)


def _parse_id_list(value: object) -> list[int]:
    """Convert a CSV cell containing comma-separated ids into a list of integers."""
    if pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]

    text = str(value).strip()
    if not text:
        return []

    return [int(item.strip()) for item in text.split(",") if item.strip()]


def load_train_test_partition(fold_id: str, metric_name: str) -> tuple[list[int], list[int]]:
    """Load test and train ids for a given fold and target metric."""
    df = pd.read_csv(FOLD_CSV_PATH)
    row = df.loc[df["fold_id"] == fold_id]
    if row.empty:
        raise ValueError(f"No fold found for fold_id '{fold_id}'.")

    test_column_name = "fold_members"
    train_column_name = f"train_set_{metric_name}"
    if train_column_name not in df.columns:
        fallback_column = f"global_complement_{metric_name}"
        if fallback_column in df.columns:
            train_column_name = fallback_column
        else:
            raise ValueError(
                f"No training fold column found for metric '{metric_name}'. "
                f"Checked: train_set_{metric_name}, global_complement_{metric_name}"
            )

    test_fold = _parse_id_list(row.iloc[0][test_column_name])
    train_fold = _parse_id_list(row.iloc[0][train_column_name])

    if len(train_fold) == 0:
        raise ValueError(f"No training fold found for test fold '{fold_id}' and metric '{metric_name}'.")

    return test_fold, train_fold


def _feature_columns_for_target(target: str) -> list[str]:
    if target == "base_load":
        return Config.XGB_FEATURES["BASE_LOAD"]
    if target == "pv_gen":
        return Config.XGB_FEATURES["PV_GEN"]
    if target in ("ev1_status", "ev2_status"):
        return Config.XGB_FEATURES["EV_STATUS"]
    raise ValueError(f"Unknown target: {target}")


def _train_df_for_target(target: str, train_fold: list[int]) -> pd.DataFrame:
    if target == "base_load":
        return get_base_load_features(train_fold)
    if target == "pv_gen":
        return get_pv_gen_features(train_fold)
    if target in ("ev1_status", "ev2_status"):
        return get_ev_status_features(train_fold)
    raise ValueError(f"Unknown target: {target}")


def _label_column_for_target(target: str) -> str:
    if target in ("base_load", "pv_gen"):
        return "next_value"
    return "next_state"


def _build_model_for_target(target: str):
    params = RIDGE_PARAMS[target]
    if target in ("base_load", "pv_gen"):
        estimator = Ridge(**params)
    else:
        estimator = RidgeClassifier(**params)

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def _save_model(model, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as fh:
        pickle.dump(model, fh)


def run() -> None:
    for target in TARGETS:
        print(f"\n=== Starting {target} ridge_regression models ===")
        print(f"  Using params: {RIDGE_PARAMS[target]}")

        for fold_id in "ABCDE":
            print(f"- Training {target} for fold {fold_id}...")
            _, train_fold = load_train_test_partition(fold_id, target)
            print(f"  Loaded {len(train_fold)} training ids for fold {fold_id}")

            train_df = _train_df_for_target(target, train_fold)
            feature_columns = _feature_columns_for_target(target)
            y_col = _label_column_for_target(target)
            X_train = train_df[feature_columns]
            y_train = train_df[y_col]

            model = _build_model_for_target(target)
            print(f"  Fitting model with {len(X_train)} rows and {len(feature_columns)} features")
            model.fit(X_train, y_train)
            print(f"  Finished training {target} model for fold {fold_id}")

            model_save_path = Path(Config.RIDGE_METRIC_MODEL_DIRS[target] / f"{fold_id}.pkl")
            _save_model(model, model_save_path)
            print(f"  Saved model to {model_save_path}")

    print("\nAll ridge_regression models trained and saved.")


if __name__ == "__main__":
    run()
