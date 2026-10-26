import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from models.train_utils import train_dataset_models  # noqa: E402


FEATURE_DIR = ROOT_DIR / "training" / "features" / "feature_parquets" / "ev2"
MODEL_DIR = ROOT_DIR / "models" / "ev2"


if __name__ == "__main__":
    train_dataset_models(
        feature_dir=FEATURE_DIR,
        file_pattern="ev2_status_features_*.parquet",
        model_dir=MODEL_DIR,
        target_column="next_state",
        learning_rate=0.1,
        max_depth=2,
        n_estimators=200,
        dataset_name="EV2",
        model_type="classification",
    )
