from __future__ import annotations

from pathlib import Path
import subprocess
import sys
# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.config import Config


def _run_training_script(script_path: Path) -> None:
    print(f"\n=== Running {script_path.name} ===")
    subprocess.run([sys.executable, str(script_path)], check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    training_dir = repo_root / "training" / "xgboost" / "training"

    print("Using Config-backed household split for all XGBoost models")
    print(f"Training households ({len(Config.H_SET_TRAINING)}): {list(Config.H_SET_TRAINING)}")
    print(f"Testing households ({len(Config.H_SET_TESTING)}): {list(Config.H_SET_TESTING)}")

    scripts = [
        training_dir / "base_load_regressor.py",
        training_dir / "pv_gen_regressor.py",
        training_dir / "ev_status_classifier.py",
    ]

    for script_path in scripts:
        if not script_path.exists():
            raise FileNotFoundError(f"Missing training script: {script_path}")
        _run_training_script(script_path)

    print("\nAll XGB models trained successfully.")


if __name__ == "__main__":
    main()