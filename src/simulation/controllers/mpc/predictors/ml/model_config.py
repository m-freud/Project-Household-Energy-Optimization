from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MODEL_TARGETS: tuple[str, ...] = (
    "base_load",
    "pv_gen",
    "ev1_status",
    "ev2_status",
)

# Backward-compatible alias.
MODEL_METRICS = MODEL_TARGETS


MODEL_FEATURES: dict[str, list[str]] = {
    "EV_STATUS": [
        "timestep",
        "status",
        "time_sin",
        "time_cos",
        "steps_in_current_state",
        "phase_id",
        "status_lag_1",
        "status_lag_1_is_pad",
        "status_lag_2",
        "status_lag_2_is_pad",
        "status_lag_4",
        "status_lag_4_is_pad",
        "status_lag_8",
        "status_lag_8_is_pad",
        "start1_earliest",
        "end1_latest",
        "start2_earliest",
        "end2_latest",
        "max_commute_steps_1",
        "max_commute_steps_2",
        "steps_to_start1_earliest",
        "steps_to_end1_latest",
        "steps_to_start2_earliest",
        "steps_to_end2_latest",
        "start1",
        "end1",
        "start2",
        "end2",
        "start1_observed",
        "end1_observed",
        "start2_observed",
        "end2_observed",
        "observed_window_length_1",
        "observed_window_length_2",
        "window_length_slack_1",
        "window_length_slack_2",
    ],
    "BASE_LOAD": [
        "timestep",
        "base_load",
        "n_evs_at_home",
        "time_sin",
        "time_cos",
        "base_load_lag_1",
        "base_load_lag_1_is_pad",
        "base_load_lag_2",
        "base_load_lag_2_is_pad",
        "base_load_lag_4",
        "base_load_lag_4_is_pad",
        "base_load_lag_8",
        "base_load_lag_8_is_pad",
        "base_load_lag_12",
        "base_load_lag_12_is_pad",
        "base_load_ma_2",
        "base_load_ma_4",
        "base_load_ma_8",
        "base_load_ma_16",
        "base_load_std_4",
        "base_load_std_8",
        "base_load_delta_1",
        "base_load_delta_2",
        "base_load_accel",
    ],
    "PV_GEN": [
        "timestep",
        "pv_gen",
        "time_sin",
        "time_cos",
        "pv_lag_1",
        "pv_lag_1_is_pad",
        "pv_lag_2",
        "pv_lag_2_is_pad",
        "pv_lag_4",
        "pv_lag_4_is_pad",
        "pv_lag_8",
        "pv_lag_8_is_pad",
        "pv_lag_12",
        "pv_lag_12_is_pad",
        "pv_ma_2",
        "pv_ma_4",
        "pv_ma_8",
        "pv_ma_16",
        "pv_std_4",
        "pv_std_8",
        "pv_delta_1",
        "pv_delta_2",
        "pv_accel",
        "steps_to_daylight_start",
        "steps_to_daylight_end",
    ],
}


@dataclass(frozen=True)
class ModelFamilyConfig:
    name: str
    model_dir: Path
    file_suffix: str
    target_model_dirs: dict[str, Path]

    def get_model_path_for_fold(self, target: str, fold_id: str) -> Path:
        target_key = str(target)
        if target_key not in self.target_model_dirs:
            valid_targets = ", ".join(sorted(self.target_model_dirs.keys()))
            raise ValueError(
                f"Unsupported target '{target_key}' for model family '{self.name}'. Expected one of: {valid_targets}"
            )
        return Path(self.target_model_dirs[target_key] / f"{fold_id}{self.file_suffix}")


def _build_target_dirs(model_dir: Path) -> dict[str, Path]:
    return {target: Path(model_dir / target) for target in MODEL_TARGETS}


def build_model_family_configs(root_dir: Path) -> dict[str, ModelFamilyConfig]:
    xgb_dir = Path(root_dir / "training" / "xgboost" / "models")
    rf_dir = Path(root_dir / "training" / "random_forest" / "models")
    ridge_dir = Path(root_dir / "training" / "ridge_regression" / "models")

    return {
        "xgb": ModelFamilyConfig(
            name="xgb",
            model_dir=xgb_dir,
            file_suffix=".json",
            target_model_dirs=_build_target_dirs(xgb_dir),
        ),
        "rf": ModelFamilyConfig(
            name="rf",
            model_dir=rf_dir,
            file_suffix=".pkl",
            target_model_dirs=_build_target_dirs(rf_dir),
        ),
        "ridge": ModelFamilyConfig(
            name="ridge",
            model_dir=ridge_dir,
            file_suffix=".pkl",
            target_model_dirs=_build_target_dirs(ridge_dir),
        ),
    }
