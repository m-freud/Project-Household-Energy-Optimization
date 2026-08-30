'''
here we try to find good hypams for any target/model
for training on portugal data
(ch data elsewhere)

procedure:
- train on inner train partition (yes, full partition)
- validate on inner validation partition
- save hypams/score to csv

'''
# paste this to enable src. imports
import itertools
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

import argparse  # noqa
import json  # noqa
import pandas as pd  # noqa
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa
from sklearn.linear_model import Ridge, RidgeClassifier  # noqa
from sklearn.metrics import root_mean_squared_error  # noqa
from sklearn.pipeline import Pipeline  # noqa
from sklearn.preprocessing import StandardScaler  # noqa
from xgboost import XGBClassifier, XGBRegressor  # noqa

from src.simulation.controllers.mpc.predictors.ml.model_config import MODEL_FEATURES_BY_FAMILY  # noqa
from training.features.base_load_features import get_base_load_features  # noqa
from training.features.ev_status_features import get_ev_status_features  # noqa
from training.features.pv_gen_features import get_pv_gen_features  # noqa
from training.split.clean_split import PARTITIONS  # noqa

GRID_MAP = {
    "ridge": {
        "grid_1": {
            "alpha": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0],
        },
        "grid_A": { # grid 1 but used with xgb features
            "alpha": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0],
        },
    },
    "xgboost": {
        "grid_1": {
            "n_estimators": [100, 200, 300, 400, 500, 600],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5],
        },
        "pv_1": {
            "n_estimators": [100, 200, 300, 400, 500, 600],
            "max_depth": [2, 3],
            "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
        },
        "ev_1": {
            "n_estimators": [100, 200, 300, 400, 500, 600],
            "max_depth": [2,3,4],
            "learning_rate": [0.01, 0.02, 0.05, 0.1],
        },
        "grid_2": {
            "n_estimators": [100, 200, 300, 400, 500, 600],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1, 0.2],
        },
        "load_1": {
            "n_estimators": [100, 200, 300, 400, 500, 600],
            "max_depth": [2, 3, 4, 5, 6],
            "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1, 0.2],
        },
        "load_2": {
            "n_estimators": [100, 200, 300, 400, 500, 600],
            "max_depth": [2, 3],
            "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.1],
        },

    },
    "random_forest": {
        "grid_1": {
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", 0.5, 1.0],
        },
        "pv_1": {
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [3, 4, 5],
            "max_features": ["sqrt"],
        },
        "ev_1": {
            "n_estimators": [100, 300, 500],
            "max_depth": [10],
            "min_samples_leaf": [2, 3, 5],
            "max_features": ["sqrt", 0.5, 1.0],
        },
        "load_1": {
            "n_estimators": [100, 300, 500],
            "max_depth": [10],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt"],
        },
        "grid_A": { # grid 1 but used with xgb features
            "n_estimators": [100, 300, 500],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", 0.5, 1.0],
        },
        "grid_2": {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [2, 3, 5],
            "max_features": ["sqrt", 0.3, 0.5, 0.7],
        },
        "grid_3": {
            "n_estimators": [100, 200],
            "max_depth": [15, 20, 25],
            "min_samples_leaf": [2, 3, 4, 5],
            "max_features": ["sqrt", 0.2, 0.3, 0.4],
        }
    },
}

OUTPUT_DIR = Path(__file__).parent


def build_param_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*grid.values())]


def get_train_test_frames(target, model_family) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    train_ids = PARTITIONS["inner"][target]["train"]
    test_ids = PARTITIONS["inner"][target]["test"]

    if target == "base_load":
        train_df = get_base_load_features(train_ids)
        test_df = get_base_load_features(test_ids)
        y_col = "next_value"
    elif target == "pv_gen":
        train_df = get_pv_gen_features(train_ids)
        test_df = get_pv_gen_features(test_ids)
        y_col = "next_value"
    elif target in ("ev1_status", "ev2_status"):
        train_df = get_ev_status_features(train_ids)
        test_df = get_ev_status_features(test_ids)
        y_col = "next_state"
    else:
        raise ValueError(f"Unknown target: {target}")

    if "ev" in target:
        feature_columns = MODEL_FEATURES_BY_FAMILY[model_family]["ev_status"]
    else:
        feature_columns = MODEL_FEATURES_BY_FAMILY[model_family][target]
    return train_df, test_df, feature_columns, y_col


def build_model(model_family: str, target: str, params: dict):
    is_regression = target in ("base_load", "pv_gen")

    if model_family == "ridge":
        estimator = Ridge(**params) if is_regression else RidgeClassifier(**params)
        # only ridge gets a scaler in the pipeline
        return Pipeline(steps=[("scaler", StandardScaler()), ("model", estimator)])

    if model_family in ["xgboost", "xgb"]:
        return XGBRegressor(**params) if is_regression else XGBClassifier(**params)

    if model_family in ["random_forest", "rf"]:
        return RandomForestRegressor(**params) if is_regression else RandomForestClassifier(**params)

    raise ValueError(f"Unknown model family: {model_family}")


def score_model(model, target: str, X_test, y_test) -> float:
    if target in ("base_load", "pv_gen"):
        return float(root_mean_squared_error(y_test, model.predict(X_test)))

    accuracy = float(model.score(X_test, y_test))
    return float(1.0 - accuracy)


def hypam_sweep(target: str, model: str, grid: str):
    '''do a hypam sweep by training models for all grid configs and saving the results to a csv file'''

    train_df, test_df, feature_columns, y_col = get_train_test_frames(target=target, model_family=model)

    X_train = train_df[feature_columns]
    y_train = train_df[y_col]
    X_test = test_df[feature_columns]
    y_test = test_df[y_col]

    out_dir = OUTPUT_DIR / "prediction" / target / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{grid}.csv"
    json_path = out_dir / f"{grid}.json"

    print(f"Running hyperparameter sweep for target: {target}, model: {model}, grid: {grid}")

    rows = []
    param_configs = build_param_grid(GRID_MAP[model][grid])
    for i, params in enumerate(param_configs, 1):
        estimator = build_model(model, target, params)
        estimator.fit(X_train, y_train)
        score = score_model(estimator, target, X_test, y_test)

        print(f"  [{i:3d}/{len(param_configs)}] params={params} -> score={score:.5f}")
        rows.append({"params": json.dumps(params, sort_keys=True), "score": round(score, 5)})

    pd.DataFrame(rows).to_csv(out_path, index=False)
    with open(json_path, "w") as f:
        json.dump(GRID_MAP[model][grid], f, indent=2)
    print(f"Saved sweep results to {out_path}")
    print(f"Saved hypam grid to {json_path}")


def normalize_target(target: str) -> str:
    return {"pv": "pv_gen", "load": "base_load", "ev1": "ev1_status", "ev2": "ev2_status"}.get(target, target)


def normalize_model(model: str) -> str:
    return {"rf": "random_forest", "xgb": "xgboost"}.get(model, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        nargs="+",
        default=["pv_gen"],
        help="target(s). choose between base_load,pv_gen,ev1_status,ev2_status",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["xgboost"],
        help="inference model(s). choose between xgboost, random_forest, ridge",
    )
    parser.add_argument(
        "--grid",
        type=str,
        nargs="+",
        default=["grid_1"],
        help="grid(s)",
    )
    args = parser.parse_args()

    targets = [normalize_target(t) for t in args.target]
    models = [normalize_model(m) for m in args.model]
    grids = args.grid

    for model in models:
        for target in targets:
            for grid in grids:
                if grid not in GRID_MAP.get(model, {}):
                    print(f"Skipping {target}/{model}/{grid}: grid not defined for model")
                    continue
                hypam_sweep(target=target, model=model, grid=grid)



