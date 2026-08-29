from pathlib import Path
import sys
import itertools
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import root_mean_squared_error, log_loss

repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.runtime_config import RuntimeConfig
from training._archive.xgboost.feature_analysis import get_base_load_features, get_pv_gen_features, get_ev_status_features
from training._archive.xgboost.training.train_models import load_train_test_partition

FOLD_IDS = list("ABCDE")

TARGETS = ["base_load", "pv_gen", "ev1_status", "ev2_status"]

HYPAM_GRID = {
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators":  [100, 300, 600],
    "max_depth":     [3, 5, 7],
}

OUTPUT_PATH = Path(repo_root) / "training" / "xgboost" / "tuning" / "results.csv"


def _build_param_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*grid.values())]


def _evaluate_fold(target: str, fold_id: str, params: dict) -> float:
    test_fold, train_fold = load_train_test_partition(fold_id, target)

    if target == "base_load":
        train_df = get_base_load_features(train_fold)
        test_df  = get_base_load_features(test_fold)
        feature_columns = RuntimeConfig.XGB_FEATURES["BASE_LOAD"]
        X_train, y_train = train_df[feature_columns], train_df["next_value"]
        X_test,  y_test  = test_df[feature_columns],  test_df["next_value"]
        model = XGBRegressor(**params, verbosity=0)
        model.fit(X_train, y_train)
        return float(root_mean_squared_error(y_test, model.predict(X_test)))

    elif target == "pv_gen":
        train_df = get_pv_gen_features(train_fold)
        test_df  = get_pv_gen_features(test_fold)
        feature_columns = RuntimeConfig.XGB_FEATURES["PV_GEN"]
        X_train, y_train = train_df[feature_columns], train_df["next_value"]
        X_test,  y_test  = test_df[feature_columns],  test_df["next_value"]
        model = XGBRegressor(**params, verbosity=0)
        model.fit(X_train, y_train)
        return float(root_mean_squared_error(y_test, model.predict(X_test)))

    elif target in ("ev1_status", "ev2_status"):
        train_df = get_ev_status_features(train_fold)
        test_df  = get_ev_status_features(test_fold)
        feature_columns = RuntimeConfig.XGB_FEATURES["EV_STATUS"]
        X_train, y_train = train_df[feature_columns], train_df["next_state"]
        X_test,  y_test  = test_df[feature_columns],  test_df["next_state"]
        model = XGBClassifier(**params, verbosity=0)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        return float(log_loss(y_test, proba))

    raise ValueError(f"Unknown target: {target}")


def tune(targets: list[str] = TARGETS, grid: dict = HYPAM_GRID) -> pd.DataFrame:
    param_configs = _build_param_grid(grid)
    rows = []

    for target in targets:
        print(f"\n=== Tuning {target} ({len(param_configs)} configs × {len(FOLD_IDS)} folds) ===")
        for i, params in enumerate(param_configs, 1):
            fold_scores = []
            for fold_id in FOLD_IDS:
                score = _evaluate_fold(target, fold_id, params)
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores))
            print(
                f"  [{i:3d}/{len(param_configs)}] lr={params['learning_rate']} "
                f"n={params['n_estimators']} depth={params['max_depth']} "
                f"→ mean={mean_score:.4f}  folds={[round(s,4) for s in fold_scores]}"
            )
            rows.append({
                "target": target,
                **params,
                "mean_score": mean_score,
                **{f"fold_{fid}": s for fid, s in zip(FOLD_IDS, fold_scores)},
            })

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
        metric = "log_loss" if target in ("ev1_status", "ev2_status") else "rmse"
        print(
            f"  {target}: lr={best['learning_rate']} n={int(best['n_estimators'])} "
            f"depth={int(best['max_depth'])}  {metric}={best['mean_score']:.4f}"
        )
