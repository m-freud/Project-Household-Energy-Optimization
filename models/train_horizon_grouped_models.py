import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from models.train_utils import train_horizon_grouped_dataset_models  # noqa: E402


TRAINING_CONFIGS = (
    {
        "target": "load",
        "file_pattern": "ch_load_features_*.parquet",
        "target_column": "next_value",
        "learning_rate": 0.02,
        "max_depth": 3,
        "n_estimators": 100,
        "dataset_name": "Load",
        "model_type": "regression",
    },
    {
        "target": "pv",
        "file_pattern": "de_pv_features_*.parquet",
        "target_column": "next_value",
        "learning_rate": 0.05,
        "max_depth": 3,
        "n_estimators": 100,
        "dataset_name": "PV",
        "model_type": "regression",
    },
    {
        "target": "ev1",
        "file_pattern": "ev1_status_features_*.parquet",
        "target_column": "next_state",
        "learning_rate": 0.1,
        "max_depth": 2,
        "n_estimators": 200,
        "dataset_name": "EV1",
        "model_type": "classification",
    },
    {
        "target": "ev2",
        "file_pattern": "ev2_status_features_*.parquet",
        "target_column": "next_state",
        "learning_rate": 0.1,
        "max_depth": 2,
        "n_estimators": 200,
        "dataset_name": "EV2",
        "model_type": "classification",
    },
)


if __name__ == "__main__":
    for config in TRAINING_CONFIGS:
        target = config["target"]
        train_horizon_grouped_dataset_models(
            feature_dir=ROOT_DIR / "training" / "features" / "feature_parquets" / target,
            file_pattern=config["file_pattern"],
            model_dir=ROOT_DIR / "models" / target,
            target_column=config["target_column"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            n_estimators=config["n_estimators"],
            dataset_name=config["dataset_name"],
            model_type=config["model_type"],
        )
