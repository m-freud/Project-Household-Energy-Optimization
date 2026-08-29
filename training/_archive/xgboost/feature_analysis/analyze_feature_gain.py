from pathlib import Path
import sys
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.runtime_config import RuntimeConfig
from training._archive.xgboost.feature_analysis import get_base_load_features, get_pv_gen_features, get_ev_status_features

FOLD_CSV_PATH = Path(RuntimeConfig.ROOT_DIR / "training" / "split" / "test_folds.csv")

FOLD_IDS = list("ABCDE")
TARGETS = ["base_load", "pv_gen", "ev1_status", "ev2_status"]
EXCLUDED_FOLDS_BY_TARGET = {
    "pv_gen": {"E"},
}

# Best params from tuning results.
BEST_PARAMS = {
    "base_load": {"learning_rate": 0.05, "n_estimators": 100, "max_depth": 3},
    "pv_gen": {"learning_rate": 0.05, "n_estimators": 600, "max_depth": 5},
    "ev1_status": {"learning_rate": 0.1, "n_estimators": 300, "max_depth": 3},
    "ev2_status": {"learning_rate": 0.1, "n_estimators": 300, "max_depth": 3},
}

OUTPUT_DIR = Path(repo_root) / "training" / "xgboost" / "features" / "importance_reports"


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
    params = BEST_PARAMS[target]
    if target in ("base_load", "pv_gen"):
        return XGBRegressor(**params, verbosity=0)
    return XGBClassifier(**params, verbosity=0)


def _effective_fold_ids(target: str) -> list[str]:
    excluded = EXCLUDED_FOLDS_BY_TARGET.get(target, set())
    return [fold_id for fold_id in FOLD_IDS if fold_id not in excluded]


def _extract_gain_importance(model, feature_columns: list[str]) -> pd.DataFrame:
    booster = model.get_booster()
    gain_by_f = booster.get_score(importance_type="gain")
    total_gain_by_f = booster.get_score(importance_type="total_gain")
    weight_by_f = booster.get_score(importance_type="weight")
    rows = []

    for idx, feature_name in enumerate(feature_columns):
        f_idx = f"f{idx}"
        gain = float(gain_by_f.get(feature_name, gain_by_f.get(f_idx, 0.0)))
        total_gain = float(total_gain_by_f.get(feature_name, total_gain_by_f.get(f_idx, 0.0)))
        split_count = float(weight_by_f.get(feature_name, weight_by_f.get(f_idx, 0.0)))
        rows.append({"feature": feature_name, "gain": gain, "total_gain": total_gain, "split_count": split_count})

    return pd.DataFrame(rows)


def analyze_target(target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_columns = _feature_columns_for_target(target)
    per_fold_frames = []
    effective_folds = _effective_fold_ids(target)

    print(f"\n=== Feature importance for {target} ===")
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

        fold_df = _extract_gain_importance(model, feature_columns)
        fold_df["target"] = target
        fold_df["fold_id"] = fold_id
        per_fold_frames.append(fold_df)

        top3 = fold_df.sort_values("total_gain", ascending=False).head(3)
        top3_msg = ", ".join([f"{r.feature}={r.total_gain:.4f}" for r in top3.itertuples(index=False)])
        print(f"  fold {fold_id}: top3 total_gain -> {top3_msg}")

    per_fold = pd.concat(per_fold_frames, ignore_index=True)

    summary = (
        per_fold.groupby(["target", "feature"], as_index=False)
        .agg(
            mean_gain=("gain", "mean"),
            std_gain=("gain", "std"),
            min_gain=("gain", "min"),
            max_gain=("gain", "max"),
            mean_total_gain=("total_gain", "mean"),
            std_total_gain=("total_gain", "std"),
            min_total_gain=("total_gain", "min"),
            max_total_gain=("total_gain", "max"),
            mean_split_count=("split_count", "mean"),
            nonzero_fold_count=("gain", lambda s: int((s > 0).sum())),
        )
        .sort_values("mean_total_gain", ascending=False)
    )
    summary["std_gain"] = summary["std_gain"].fillna(0.0)
    summary["std_total_gain"] = summary["std_total_gain"].fillna(0.0)

    max_mean_total_gain = float(summary["mean_total_gain"].max()) if len(summary) else 0.0
    weak_threshold = 0.02 * max_mean_total_gain if max_mean_total_gain > 0 else 0.0

    drop_candidates = summary[
        (summary["mean_total_gain"] <= weak_threshold) & (summary["nonzero_fold_count"] <= 2)
    ].copy()
    drop_candidates.insert(
        1,
        "rule",
        f"mean_total_gain <= 2% of max_mean_total_gain ({weak_threshold:.6f}) and nonzero_fold_count <= 2",
    )

    return per_fold, summary, drop_candidates


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_per_fold = []
    all_summary = []
    all_drop_candidates = []

    for target in TARGETS:
        per_fold, summary, drop_candidates = analyze_target(target)
        all_per_fold.append(per_fold)
        all_summary.append(summary)
        all_drop_candidates.append(drop_candidates)

        per_fold_path = OUTPUT_DIR / f"{target}_gain_per_fold.csv"
        summary_path = OUTPUT_DIR / f"{target}_gain_summary.csv"
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

    merged_per_fold.to_csv(OUTPUT_DIR / "all_targets_gain_per_fold.csv", index=False)
    merged_summary.to_csv(OUTPUT_DIR / "all_targets_gain_summary.csv", index=False)
    merged_drop.to_csv(OUTPUT_DIR / "all_targets_drop_candidates.csv", index=False)

    print("\n=== Done ===")
    print(f"Reports directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
