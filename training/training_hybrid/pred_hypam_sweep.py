"""Sweep fixed XGBoost models trained on mixed CH and Portugal base-load days."""
from pathlib import Path
import json
import sys

# Find the repository root that contains 'src'.
repo_root = next((path for path in Path.cwd().resolve().parents if (path / "src").exists()), "")
sys.path.insert(0, str(repo_root))

import pandas as pd
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

from src.simulation.controllers.mpc.predictors.ml.model_config import MODEL_FEATURES_BY_FAMILY
from training.features.base_load_features import get_base_load_features
from training.split.clean_split import PARTITIONS
from training.split.equality_groups import eg_base_load, select_balanced_group_members
from training.training_ch_load.tuning.pred_hypam_sweep import load_feature_sample_df


OUTPUT_DIR = Path(__file__).parent / "pred_cross_data_sweep"
MODEL_PARAMS = {"learning_rate": 0.03, "max_depth": 3, "n_estimators": 200}
TOTAL_PLAYER_DAYS = 100
DATE_STEPS = 3
MIXES = [(0, 100), (20, 80), (40, 60), (50, 50), (60, 40), (80, 20), (100, 0)]


def run_cross_data_sweep() -> pd.DataFrame:
    features = MODEL_FEATURES_BY_FAMILY["xgboost"]["base_load"]
    portugal_train_ids = list(PARTITIONS["global"]["train"])
    portugal_test_df = get_base_load_features(PARTITIONS["global"]["test"])
    X_test = portugal_test_df[features].to_numpy()
    y_test = portugal_test_df["next_value"].to_numpy()
    ch_train_df = load_feature_sample_df(
        start_id=105366,
        n_days=TOTAL_PLAYER_DAYS,
        model_family="xgboost",
        date_steps=DATE_STEPS,
    )

    rows = []
    selections = {}
    for ch_days, portugal_days in MIXES:
        portugal_ids = select_balanced_group_members(
            eg_base_load, portugal_train_ids, portugal_days
        )
        if portugal_ids:
            portugal_train_df = get_base_load_features(portugal_ids)[features + ["next_value"]]
        else:
            portugal_train_df = pd.DataFrame(columns=features + ["next_value"])
        train_df = pd.concat(
            [ch_train_df.iloc[: ch_days * 96], portugal_train_df], ignore_index=True
        )
        expected_rows = TOTAL_PLAYER_DAYS * 96
        if len(train_df) != expected_rows:
            raise ValueError(f"Expected {expected_rows} training rows, got {len(train_df)}")

        model = XGBRegressor(**MODEL_PARAMS)
        model.fit(train_df[features].to_numpy(), train_df["next_value"].to_numpy())
        rmse = float(root_mean_squared_error(y_test, model.predict(X_test)))
        print(f"CH={ch_days:3d}, Portugal={portugal_days:3d} -> rmse={rmse:.5f}")
        rows.append(
            {
                "ch_player_days": ch_days,
                "portugal_player_days": portugal_days,
                "total_player_days": TOTAL_PLAYER_DAYS,
                "score": round(rmse, 5),
            }
        )
        selections[f"ch_{ch_days}_portugal_{portugal_days}"] = portugal_ids

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "base_load_xgboost_global_105366.csv"
    metadata_path = OUTPUT_DIR / "base_load_xgboost_global_105366.json"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    with open(metadata_path, "w") as file:
        json.dump(
            {
                "model": "xgboost",
                "params": MODEL_PARAMS,
                "total_player_days": TOTAL_PLAYER_DAYS,
                "ch_sample": "diag_105366_features.parquet",
                "ch_date_steps": DATE_STEPS,
                "portugal_train_set": "PARTITIONS.global.train",
                "portugal_test_set": "PARTITIONS.global.test",
                "mixes": [
                    {"ch_player_days": ch_days, "portugal_player_days": portugal_days}
                    for ch_days, portugal_days in MIXES
                ],
                "portugal_train_id_selections": selections,
            },
            file,
            indent=2,
        )
    print(f"Saved sweep results to {result_path}")
    print(f"Saved sweep metadata to {metadata_path}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_cross_data_sweep()