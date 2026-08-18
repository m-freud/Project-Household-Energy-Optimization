from pathlib import Path
import pickle
import sys
# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))
import pandas as pd
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from runtime_config import RuntimeConfig
from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig
from training._features.base_load_features import get_base_load_features
from training._features.pv_gen_features import get_pv_gen_features
from training._features.ev_status_features import get_ev_status_features
from training.model_artifacts import write_training_params_manifest

repo_root = RuntimeConfig.ROOT_DIR

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


def load_train_test_partition(fold_id: str, metric_name: str) -> tuple[list[int], list[int]]:
    """Load test and train ids for a given fold and target metric from Config."""
    test_fold = RuntimeConfig.get_fold_members(metric_name, fold_id)
    train_fold = RuntimeConfig.get_training_ids_for_fold(metric_name, fold_id)
    return test_fold, train_fold


def _feature_columns_for_target(target: str) -> list[str]:
    family_features = ModelConfig.MODEL_FEATURES_BY_FAMILY["ridge"]
    if target == "base_load":
        return family_features["base_load"]
    if target == "pv_gen":
        return family_features["pv_gen"]
    if target in ("ev1_status", "ev2_status"):
        return family_features["ev_status"]
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

        for fold_id in RuntimeConfig.FOLD_IDS:
            print(f"- Training {target} for fold {fold_id}...")
            _, train_fold = load_train_test_partition(fold_id, target)
            print(f"  Loaded {len(train_fold)} training ids for fold {fold_id}")

            train_df = _train_df_for_target(target, train_fold)
            feature_columns = _feature_columns_for_target(target)
            y_col = _label_column_for_target(target)
            X_train = train_df[feature_columns].to_numpy()
            y_train = train_df[y_col]

            model = _build_model_for_target(target)
            print(f"  Fitting model with {len(X_train)} rows and {len(feature_columns)} features")
            model.fit(X_train, y_train)
            print(f"  Finished training {target} model for fold {fold_id}")

            model_save_path = Path(RuntimeConfig.RIDGE_METRIC_MODEL_DIRS[target] / f"{fold_id}.pkl")
            _save_model(model, model_save_path)
            write_training_params_manifest(
                model_save_path.parent,
                family="ridge",
                target=target,
                fold_ids=RuntimeConfig.FOLD_IDS,
                params=RIDGE_PARAMS[target],
            )
            print(f"  Saved model to {model_save_path}")

    print("\nAll ridge_regression models trained and saved.")


if __name__ == "__main__":
    run()
