from pathlib import Path
import sys

import pandas as pd

# paste this to enable src. imports
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.runtime_config import RuntimeConfig

XGB_SUMMARY_PATH = Path(
    RuntimeConfig.ROOT_DIR / "training" / "xgboost" / "features" / "importance_reports" / "all_targets_gain_summary.csv"
)
RF_SUMMARY_PATH = Path(
    RuntimeConfig.ROOT_DIR
    / "training"
    / "random_forest"
    / "features"
    / "importance_reports"
    / "all_targets_importance_summary.csv"
)
RIDGE_SUMMARY_PATH = Path(
    RuntimeConfig.ROOT_DIR
    / "training"
    / "ridge_regression"
    / "features"
    / "importance_reports"
    / "all_targets_coef_summary.csv"
)

OUTPUT_DIR = Path(RuntimeConfig.ROOT_DIR / "training" / "_features" / "relevance_comparison_reports")
TARGETS = ["base_load", "pv_gen", "ev1_status", "ev2_status"]


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            "Missing required report file: "
            f"{path}\n"
            "Generate reports first using:\n"
            "- training/xgboost/features/analyze_feature_gain.py\n"
            "- training/random_forest/features/analyze_feature_importance.py\n"
            "- training/ridge_regression/features/analyze_feature_importance.py"
        )


def _safe_normalize(series: pd.Series) -> pd.Series:
    max_value = float(series.max()) if len(series) else 0.0
    if max_value <= 0:
        return pd.Series(0.0, index=series.index)
    return series / max_value


def _prepare_xgb() -> pd.DataFrame:
    _require_file(XGB_SUMMARY_PATH)
    xgb = pd.read_csv(XGB_SUMMARY_PATH)

    required = {"target", "feature", "mean_gain", "mean_total_gain", "nonzero_fold_count"}
    missing = required.difference(xgb.columns)
    if missing:
        raise ValueError(f"XGB summary missing columns: {sorted(missing)}")

    out = xgb[["target", "feature", "mean_gain", "mean_total_gain", "nonzero_fold_count"]].copy()
    out = out.rename(
        columns={
            "mean_gain": "xgb_mean_gain",
            "mean_total_gain": "xgb_mean_total_gain",
            "nonzero_fold_count": "xgb_nonzero_fold_count",
        }
    )
    return out


def _prepare_rf() -> pd.DataFrame:
    _require_file(RF_SUMMARY_PATH)
    rf = pd.read_csv(RF_SUMMARY_PATH)

    required = {"target", "feature", "mean_importance", "nonzero_fold_count"}
    missing = required.difference(rf.columns)
    if missing:
        raise ValueError(f"RF summary missing columns: {sorted(missing)}")

    out = rf[["target", "feature", "mean_importance", "nonzero_fold_count"]].copy()
    out = out.rename(
        columns={
            "mean_importance": "rf_mean_importance",
            "nonzero_fold_count": "rf_nonzero_fold_count",
        }
    )
    return out


def _prepare_ridge() -> pd.DataFrame:
    _require_file(RIDGE_SUMMARY_PATH)
    ridge = pd.read_csv(RIDGE_SUMMARY_PATH)

    required = {"target", "feature", "mean_abs_coef", "nonzero_fold_count"}
    missing = required.difference(ridge.columns)
    if missing:
        raise ValueError(f"Ridge summary missing columns: {sorted(missing)}")

    out = ridge[["target", "feature", "mean_abs_coef", "nonzero_fold_count"]].copy()
    out = out.rename(
        columns={
            "mean_abs_coef": "ridge_mean_abs_coef",
            "nonzero_fold_count": "ridge_nonzero_fold_count",
        }
    )
    return out


def _add_targetwise_relative_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["xgb_rel_total_gain"] = (
        df.groupby("target")["xgb_mean_total_gain"].transform(_safe_normalize).fillna(0.0)
    )
    df["rf_rel_importance"] = (
        df.groupby("target")["rf_mean_importance"].transform(_safe_normalize).fillna(0.0)
    )
    df["ridge_rel_abs_coef"] = (
        df.groupby("target")["ridge_mean_abs_coef"].transform(_safe_normalize).fillna(0.0)
    )

    return df


def _add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["xgb_rank"] = (
        df.groupby("target")["xgb_mean_total_gain"].rank(method="dense", ascending=False).astype(int)
    )
    df["rf_rank"] = (
        df.groupby("target")["rf_mean_importance"].rank(method="dense", ascending=False).astype(int)
    )
    df["ridge_rank"] = (
        df.groupby("target")["ridge_mean_abs_coef"].rank(method="dense", ascending=False).astype(int)
    )

    return df


def _add_consensus_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["mean_relative_relevance"] = (
        df[["xgb_rel_total_gain", "rf_rel_importance", "ridge_rel_abs_coef"]].mean(axis=1)
    )

    df["all_models_zero"] = (
        (df["xgb_mean_total_gain"] <= 0)
        & (df["rf_mean_importance"] <= 0)
        & (df["ridge_mean_abs_coef"] <= 0)
    )

    df["low_relevance_all_models"] = (
        (df["xgb_rel_total_gain"] <= 0.02)
        & (df["rf_rel_importance"] <= 0.02)
        & (df["ridge_rel_abs_coef"] <= 0.02)
    )

    return df


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [
        "target",
        "feature",
        "xgb_mean_gain",
        "xgb_mean_total_gain",
        "xgb_rel_total_gain",
        "xgb_rank",
        "xgb_nonzero_fold_count",
        "rf_mean_importance",
        "rf_rel_importance",
        "rf_rank",
        "rf_nonzero_fold_count",
        "ridge_mean_abs_coef",
        "ridge_rel_abs_coef",
        "ridge_rank",
        "ridge_nonzero_fold_count",
        "mean_relative_relevance",
        "all_models_zero",
        "low_relevance_all_models",
    ]
    return df[ordered]


def build_comparison_table() -> pd.DataFrame:
    xgb = _prepare_xgb()
    rf = _prepare_rf()
    ridge = _prepare_ridge()

    merged = xgb.merge(rf, on=["target", "feature"], how="outer")
    merged = merged.merge(ridge, on=["target", "feature"], how="outer")

    numeric_columns = [
        "xgb_mean_gain",
        "xgb_mean_total_gain",
        "xgb_nonzero_fold_count",
        "rf_mean_importance",
        "rf_nonzero_fold_count",
        "ridge_mean_abs_coef",
        "ridge_nonzero_fold_count",
    ]
    for col in numeric_columns:
        merged[col] = merged[col].fillna(0.0)

    merged = _add_targetwise_relative_scores(merged)
    merged = _add_ranks(merged)
    merged = _add_consensus_columns(merged)
    merged = _reorder_columns(merged)

    merged = merged.sort_values(["target", "mean_relative_relevance"], ascending=[True, False])

    return merged


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    merged = build_comparison_table()

    all_path = OUTPUT_DIR / "all_targets_feature_relevance_comparison.csv"
    merged.to_csv(all_path, index=False)
    print(f"wrote: {all_path}")

    for target in TARGETS:
        target_df = merged.loc[merged["target"] == target].copy()
        target_path = OUTPUT_DIR / f"{target}_feature_relevance_comparison.csv"
        target_df.to_csv(target_path, index=False)
        print(f"wrote: {target_path}")

    low_all = merged.loc[merged["low_relevance_all_models"]].copy()
    low_all_path = OUTPUT_DIR / "all_targets_low_relevance_all_models.csv"
    low_all.to_csv(low_all_path, index=False)
    print(f"wrote: {low_all_path}")

    print("\nDone.")
    print(f"Reports directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
