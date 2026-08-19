from pathlib import Path
import sys
import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.runtime_config import RuntimeConfig
from training._features.base_load_features import get_base_load_features
from training._features.pv_gen_features import get_pv_gen_features
from training._features.ev_status_features import get_ev_status_features
from training.ridge.training.train_models import load_train_test_partition

FOLD_IDS = list("ABCDE")
TARGETS = ["base_load", "pv_gen", "ev1_status", "ev2_status"]
EXCLUDED_FOLDS_BY_TARGET = {
    "pv_gen": {"E"},
}

HYPAM_GRID = {
    "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
}

OUTPUT_PATH = Path(repo_root) / "training" / "ridge" / "tuning" / "results.csv"


def _build_param_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*grid.values())]


def _effective_fold_ids(target: str) -> list[str]:
    excluded = EXCLUDED_FOLDS_BY_TARGET.get(target, set())
    return [fold_id for fold_id in FOLD_IDS if fold_id not in excluded]


def _build_model_for_target(target: str, params: dict):
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


def _evaluate_fold(target: str, fold_id: str, params: dict) -> float:
    test_fold, train_fold = load_train_test_partition(fold_id, target)

    if target == "base_load":
        train_df = get_base_load_features(train_fold)
        test_df = get_base_load_features(test_fold)
        feature_columns = RuntimeConfig.XGB_FEATURES["BASE_LOAD"]
        y_col = "next_value"
    elif target == "pv_gen":
        train_df = get_pv_gen_features(train_fold)
        test_df = get_pv_gen_features(test_fold)
        feature_columns = RuntimeConfig.XGB_FEATURES["PV_GEN"]
        y_col = "next_value"
    elif target in ("ev1_status", "ev2_status"):
        train_df = get_ev_status_features(train_fold)
        test_df = get_ev_status_features(test_fold)
        feature_columns = RuntimeConfig.XGB_FEATURES["EV_STATUS"]
        y_col = "next_state"
    else:
        raise ValueError(f"Unknown target: {target}")

    X_train, y_train = train_df[feature_columns], train_df[y_col]
    X_test, y_test = test_df[feature_columns], test_df[y_col]

    model = _build_model_for_target(target, params)
    model.fit(X_train, y_train)

    if target in ("base_load", "pv_gen"):
        return float(root_mean_squared_error(y_test, model.predict(X_test)))

    accuracy = float(model.score(X_test, y_test))
    return float(1.0 - accuracy)


def tune(targets: list[str] = TARGETS, grid: dict = HYPAM_GRID) -> pd.DataFrame:
    param_configs = _build_param_grid(grid)
    rows = []

    for target in targets:
        fold_ids = _effective_fold_ids(target)
        print(f"\n=== Tuning {target} ({len(param_configs)} configs x {len(fold_ids)} folds) ===")
        if len(fold_ids) != len(FOLD_IDS):
            skipped = sorted(set(FOLD_IDS).difference(fold_ids))
            print(f"  Skipping folds for {target}: {', '.join(skipped)}")

        for i, params in enumerate(param_configs, 1):
            fold_scores = []
            for fold_id in fold_ids:
                score = _evaluate_fold(target, fold_id, params)
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores))
            metric = "err" if target in ("ev1_status", "ev2_status") else "rmse"
            print(
                f"  [{i:3d}/{len(param_configs)}] alpha={params['alpha']} "
                f"-> mean_{metric}={mean_score:.4f} folds={[round(s, 4) for s in fold_scores]}"
            )

            row = {
                "target": target,
                **params,
                "mean_score": mean_score,
            }
            for fid in FOLD_IDS:
                row[f"fold_{fid}"] = np.nan
            for fid, score in zip(fold_ids, fold_scores):
                row[f"fold_{fid}"] = score
            rows.append(row)

    results = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved to {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    results = tune()

    print("\n=== Best config per target ===")
    for target, group in results.groupby("target"):
        best = group.loc[group["mean_score"].idxmin()]
        metric = "err" if target in ("ev1_status", "ev2_status") else "rmse"
        print(f"  {target}: alpha={best['alpha']} {metric}={best['mean_score']:.4f}")
