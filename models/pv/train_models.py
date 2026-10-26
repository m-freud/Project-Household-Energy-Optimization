import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from models.train_utils import train_dataset_models  # noqa: E402


FEATURE_DIR = ROOT_DIR / "training" / "features" / "feature_parquets" / "pv"
MODEL_DIR = ROOT_DIR / "models" / "pv"


if __name__ == "__main__":
    train_dataset_models(
        feature_dir=FEATURE_DIR,
        file_pattern="de_pv_features_*.parquet",
        model_dir=MODEL_DIR,
        target_column="next_value",
        learning_rate=0.05,
        max_depth=3,
        n_estimators=100,
        dataset_name="PV",
    )
