# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))



import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.runtime_config import RuntimeConfig
from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig
from training._features.base_load_features import get_base_load_features
from training._features.ev_status_features import get_ev_status_features
from training._features.pv_gen_features import get_pv_gen_features
from training.split.clean_split import GLOBAL_TEST_SET

TARGETS = ["base_load", "pv_gen", "ev1_status", "ev2_status"]
CSV_PATH = Path(RuntimeConfig.ROOT_DIR) / "training" / "ridge" / "tuning" / "tuning_results12345.csv"
OUTPUT_NAME = "best_save_model.pkl"


def _load_best_params() -> dict[str, dict]:
    results = pd.read_csv(CSV_PATH)
    if "params" not in results.columns or "score_sim_total_cost" not in results.columns:
        raise ValueError(f"Expected tuning CSV columns missing from {CSV_PATH}")

    results = results.copy()
    results["params"] = results["params"].map(json.loads)
    best_by_target: dict[str, dict] = {}

    for target in TARGETS:
        rows = results[results["target"] == target]
        if rows.empty:
            raise ValueError(f"No tuning rows found for target {target}")
        winner = rows.loc[rows["score_sim_total_cost"].idxmin()]
        best_by_target[target] = dict(winner["params"])

    return best_by_target


def _feature_frame_for_target(target: str, household_ids: list[int]) -> pd.DataFrame:
    if target == "base_load":
        return get_base_load_features(household_ids)
    if target == "pv_gen":
        return get_pv_gen_features(household_ids)
    if target in ("ev1_status", "ev2_status"):
        return get_ev_status_features(household_ids)
    raise ValueError(f"Unknown target: {target}")


def _label_column_for_target(target: str) -> str:
    if target in ("base_load", "pv_gen"):
        return "next_value"
    return "next_state"


def _feature_columns_for_target(target: str) -> list[str]:
    family_features = ModelConfig.MODEL_FEATURES_BY_FAMILY["ridge"]
    if target == "base_load":
        return family_features["base_load"]
    if target == "pv_gen":
        return family_features["pv_gen"]
    if target in ("ev1_status", "ev2_status"):
        return family_features["ev_status"]
    raise ValueError(f"Unknown target: {target}")


def _build_model(target: str, params: dict):
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


def main() -> None:
    best_params = _load_best_params()
    print("Best params by target:")
    for target in TARGETS:
        print(f"  {target}: {best_params[target]}")

    for target in TARGETS:
        train_df = _feature_frame_for_target(target, GLOBAL_TEST_SET)
        X = train_df[_feature_columns_for_target(target)].to_numpy()
        y = train_df[_label_column_for_target(target)].to_numpy()

        model = _build_model(target, best_params[target])
        model.fit(X, y)

        save_dir = RuntimeConfig.RIDGE_METRIC_MODEL_DIRS[target]
        save_path = save_dir / OUTPUT_NAME
        _save_model(model, save_path)
        print(f"Saved {target} -> {save_path}")


if __name__ == "__main__":
    main()
