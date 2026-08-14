# paste this to enable src. imports
from pathlib import Path
import sys
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig
from training.xgboost.features import get_base_load_features, get_pv_gen_features, get_ev_status_features

FOLD_CSV_PATH = Path(Config.ROOT_DIR / "training" / "split" / "test_folds.csv")
TUNING_RESULTS_CSV_PATH = Path(Config.ROOT_DIR / "training" / "xgboost" / "tuning" / "results.csv")

test_sets = Config.RUNTIME_TEST_FOLDS


def _feature_columns_for_target(target: str) -> list[str]:
    family_features = ModelConfig.MODEL_FEATURES_BY_FAMILY["xgboost"]
    if target in ("ev1_status", "ev2_status"):
        return family_features["ev_status"]
    return family_features[target]


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


def load_train_test_partition(fold_id: str, metric_name: str):
    """Load the train and test partition ids for a given fold from the CSV.
    Training folds are stored in the CSV under global_complement_{metric_name}.
    """
    df = pd.read_csv(FOLD_CSV_PATH)
    row = df.loc[df["fold_id"] == fold_id]
    if row.empty:
        raise ValueError(f"No fold found for fold_id '{fold_id}'.")

    test_column_name = "fold_members"
    train_column_name = f"train_set_{metric_name}"

    test_fold = _parse_id_list(row.iloc[0][test_column_name]) #TODO this is stupid
    train_fold = _parse_id_list(row.iloc[0][train_column_name])

    if len(train_fold) == 0:
        raise ValueError(f"No training fold found for test fold '{fold_id}' and metric '{metric_name}'.")

    return test_fold, train_fold


def load_best_params_from_tuning() -> dict[str, dict[str, float | int]]:
    if not TUNING_RESULTS_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing tuning results file: {TUNING_RESULTS_CSV_PATH}")

    results = pd.read_csv(TUNING_RESULTS_CSV_PATH)
    required_cols = {"target", "learning_rate", "n_estimators", "max_depth", "mean_score"}
    missing_cols = required_cols.difference(results.columns)
    if missing_cols:
        raise ValueError(f"Tuning results is missing required columns: {sorted(missing_cols)}")

    best_params_by_target: dict[str, dict[str, float | int]] = {}
    for target in ["base_load", "pv_gen", "ev1_status", "ev2_status"]:
        target_rows = results.loc[results["target"] == target]
        if target_rows.empty:
            raise ValueError(f"No tuning rows found for target '{target}' in {TUNING_RESULTS_CSV_PATH}")

        best_row = target_rows.loc[target_rows["mean_score"].idxmin()]
        best_params_by_target[target] = {
            "learning_rate": float(best_row["learning_rate"]),
            "n_estimators": int(best_row["n_estimators"]),
            "max_depth": int(best_row["max_depth"]),
        }

    return best_params_by_target


best_params_by_target = load_best_params_from_tuning()

for target in ["base_load", "pv_gen", "ev1_status", "ev2_status"]:
    print(f"\n=== Starting {target} models ===")
    target_params = best_params_by_target[target]
    print(
        "  Using params: "
        f"learning_rate={target_params['learning_rate']}, "
        f"n_estimators={target_params['n_estimators']}, "
        f"max_depth={target_params['max_depth']}"
    )
    for fold_id in "ABCDE":
        print(f"- Training {target} for fold {fold_id}...")
        _, train_fold = load_train_test_partition(fold_id, target)
        print(f"  Loaded {len(train_fold)} training ids for fold {fold_id}")

        if target == "base_load":
            train_df = get_base_load_features(train_fold)
            feature_columns = _feature_columns_for_target(target)
            X_train, y_train = train_df[feature_columns], train_df["next_value"]
            model = XGBRegressor(**target_params, verbosity=0)
        elif target == "pv_gen":
            train_df = get_pv_gen_features(train_fold)
            feature_columns = _feature_columns_for_target(target)
            X_train, y_train = train_df[feature_columns], train_df["next_value"]
            model = XGBRegressor(**target_params, verbosity=0)
        elif target in ["ev1_status", "ev2_status"]:
            train_df = get_ev_status_features(train_fold)
            feature_columns = _feature_columns_for_target(target)
            X_train, y_train = train_df[feature_columns], train_df["next_state"]
            model = XGBClassifier(**target_params, verbosity=0)

        print(f"  Fitting model with {len(X_train)} rows and {len(feature_columns)} features")
        model.fit(X_train, y_train)
        print(f"  Finished training {target} model for fold {fold_id}")

        model_save_path = Path(Config.XGB_METRIC_MODEL_DIRS[target] / f"{fold_id}.json")
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(model_save_path))
        print(f"  Saved model to {model_save_path}")

print("\nAll initial models trained and saved.")
        