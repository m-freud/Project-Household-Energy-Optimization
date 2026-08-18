from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.runtime_conig import RuntimeConfig
from training._features.base_load_features import get_base_load_features
from training._features.pv_gen_features import get_pv_gen_features
from training._features.ev_status_features import get_ev_status_features

FOLD_CSV_PATH = Path(RuntimeConfig.ROOT_DIR / "training" / "split" / "test_folds.csv")

FOLD_IDS = list("ABCDE")
TARGETS = ["base_load", "pv_gen", "ev1_status", "ev2_status"]
EXCLUDED_FOLDS_BY_TARGET = {
    "pv_gen": {"E"},
}

TUNING_RESULTS_CSV_PATH = Path(repo_root) / "training" / "ridge_regression" / "tuning" / "results.csv"

RIDGE_PARAMS = {
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


RIDGE_PARAMS = _load_best_params_from_tuning(RIDGE_PARAMS)

OUTPUT_DIR = Path(repo_root) / "training" / "ridge_regression" / "features" / "importance_reports"


def _parse_id_list(value: object) -> list[int]:
    if pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]

    text = str(value).strip()
    if not text:
        return []

    return [int(item.strip()) for item in text.split(",") if item.strip()]


def load_train_test_partition(fold_id: str, metric_name: str) -> tuple[list[int], list[int]]:
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
        return RuntimeConfig.XGB_FEATURES["BASE_LOAD"]
    if target == "pv_gen":
        return RuntimeConfig.XGB_FEATURES["PV_GEN"]
    if target in ("ev1_status", "ev2_status"):
        return RuntimeConfig.XGB_FEATURES["EV_STATUS"]
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


def _effective_fold_ids(target: str) -> list[str]:
    excluded = EXCLUDED_FOLDS_BY_TARGET.get(target, set())
    return [fold_id for fold_id in FOLD_IDS if fold_id not in excluded]


def _extract_abs_coefficient_importance(model: Pipeline, feature_columns: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    coef = np.asarray(estimator.coef_)

    if coef.ndim == 1:
        abs_coef = np.abs(coef)
    else:
        # Multiclass: aggregate per-feature relevance across classes.
        abs_coef = np.mean(np.abs(coef), axis=0)

    rows = []
    for feature_name, importance in zip(feature_columns, abs_coef, strict=True):
        rows.append({"feature": feature_name, "abs_coef": float(importance)})

    return pd.DataFrame(rows)


def analyze_target(target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_columns = _feature_columns_for_target(target)
    per_fold_frames = []
    effective_folds = _effective_fold_ids(target)

    print(f"\n=== Feature relevance for {target} (ridge_regression) ===")
    if len(effective_folds) != len(FOLD_IDS):
        skipped = sorted(set(FOLD_IDS).difference(effective_folds))
        print(f"  Skipping folds for {target}: {', '.join(skipped)}")

    for fold_id in effective_folds:
        _, train_fold = load_train_test_partition(fold_id, target)
        train_df = _train_df_for_target(target, train_fold)
        y_col = _label_column_for_target(target)

        X_train = train_df[feature_columns]
        y_train = train_df[y_col]

        model = _build_model_for_target(target)
        model.fit(X_train, y_train)

        fold_df = _extract_abs_coefficient_importance(model, feature_columns)
        fold_df["target"] = target
        fold_df["fold_id"] = fold_id
        per_fold_frames.append(fold_df)

        top3 = fold_df.sort_values("abs_coef", ascending=False).head(3)
        top3_msg = ", ".join([f"{r.feature}={r.abs_coef:.6f}" for r in top3.itertuples(index=False)])
        print(f"  fold {fold_id}: top3 abs_coef -> {top3_msg}")

    per_fold = pd.concat(per_fold_frames, ignore_index=True)

    summary = (
        per_fold.groupby(["target", "feature"], as_index=False)
        .agg(
            mean_abs_coef=("abs_coef", "mean"),
            std_abs_coef=("abs_coef", "std"),
            min_abs_coef=("abs_coef", "min"),
            max_abs_coef=("abs_coef", "max"),
            nonzero_fold_count=("abs_coef", lambda s: int((s > 0).sum())),
        )
        .sort_values("mean_abs_coef", ascending=False)
    )
    summary["std_abs_coef"] = summary["std_abs_coef"].fillna(0.0)

    max_mean_abs_coef = float(summary["mean_abs_coef"].max()) if len(summary) else 0.0
    weak_threshold = 0.02 * max_mean_abs_coef if max_mean_abs_coef > 0 else 0.0

    drop_candidates = summary[
        (summary["mean_abs_coef"] <= weak_threshold) & (summary["nonzero_fold_count"] <= 2)
    ].copy()
    drop_candidates.insert(
        1,
        "rule",
        f"mean_abs_coef <= 2% of max_mean_abs_coef ({weak_threshold:.6f}) and nonzero_fold_count <= 2",
    )

    return per_fold, summary, drop_candidates


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_per_fold = []
    all_summary = []
    all_drop_candidates = []

    for target in TARGETS:
        per_fold, summary, drop_candidates = analyze_target(target)
        all_per_fold.append(per_fold)
        all_summary.append(summary)
        all_drop_candidates.append(drop_candidates)

        per_fold_path = OUTPUT_DIR / f"{target}_coef_per_fold.csv"
        summary_path = OUTPUT_DIR / f"{target}_coef_summary.csv"
        drop_path = OUTPUT_DIR / f"{target}_drop_candidates.csv"

        per_fold.to_csv(per_fold_path, index=False)
        summary.to_csv(summary_path, index=False)
        drop_candidates.to_csv(drop_path, index=False)

        print(f"  wrote: {per_fold_path}")
        print(f"  wrote: {summary_path}")
        print(f"  wrote: {drop_path}")

    merged_per_fold = pd.concat(all_per_fold, ignore_index=True)
    merged_summary = pd.concat(all_summary, ignore_index=True)
    merged_drop = pd.concat(all_drop_candidates, ignore_index=True)

    merged_per_fold.to_csv(OUTPUT_DIR / "all_targets_coef_per_fold.csv", index=False)
    merged_summary.to_csv(OUTPUT_DIR / "all_targets_coef_summary.csv", index=False)
    merged_drop.to_csv(OUTPUT_DIR / "all_targets_drop_candidates.csv", index=False)

    print("\n=== Done ===")
    print(f"Reports directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
