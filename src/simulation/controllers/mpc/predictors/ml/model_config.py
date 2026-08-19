from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.simulation.controllers.mpc.predictors.ml.model_interface import ModelLike


MODEL_TARGETS: list[str] = [
    "base_load",
    "pv_gen",
    "ev1_status",
    "ev2_status",
]


MODEL_FEATURE_DOMAIN: dict[str, list[str]] = {
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


MODEL_FEATURES_BY_FAMILY: dict[str, dict[str, list[str]]] = {
    "xgboost": {
        "ev_status": [
            "status",
            "steps_in_current_state",
            "status_lag_1",
            "status_lag_2",
            "time_cos",
            "start2",
            "phase_id",
            "status_lag_4",
            "timestep",
            "window_length_slack_1",
            "end1",
            "time_sin",
            "start1",
            "observed_window_length_1",
        ],
        "base_load": [
            "base_load",
            "base_load_ma_8",
            "base_load_ma_2",
            "base_load_lag_12",
            "base_load_ma_4",
            "base_load_ma_16",
            "base_load_lag_8",
            "time_sin",
            "base_load_lag_2",
            "timestep",
            "base_load_accel",
            "base_load_delta_1",
            "base_load_lag_4",
            "time_cos",
            "base_load_std_4",
            "base_load_std_8",
            "base_load_delta_2",
            "base_load_lag_1",
            "n_evs_at_home",
        ],
        "pv_gen": [
            "pv_gen",
            "pv_ma_2",
            "timestep",
            "pv_lag_1",
            "pv_lag_12",
            "time_cos",
            "pv_delta_1",
            "pv_lag_8",
            "pv_std_8",
            "pv_delta_2",
            "pv_lag_4",
            "pv_accel",
            "pv_std_4",
            "pv_ma_8",
            "pv_lag_2",
            "pv_ma_16",
            "pv_ma_4",
            "time_sin",
        ],
    },
    "random_forest": {
        "ev_status": [
            "status",
            "status_lag_2",
            "status_lag_1",
            "phase_id",
            "status_lag_4",
            "start1_observed",
            "start2",
            "start1",
            "start2_observed",
            "steps_to_start1_earliest",
            "steps_to_end1_latest",
            "timestep",
            "steps_to_start2_earliest",
            "steps_to_end2_latest",
            "steps_in_current_state",
            "window_length_slack_2",
            "time_cos",
            "end2_observed",
            "observed_window_length_2",
            "end1",
            "end2",
            "end1_observed",
            "observed_window_length_1",
            "window_length_slack_1",
            "status_lag_8",
            "time_sin",
        ],
        "base_load": [
            "base_load",
            "base_load_ma_2",
            "base_load_ma_4",
            "base_load_ma_8",
            "base_load_accel",
            "base_load_delta_1",
            "base_load_lag_1",
            "base_load_ma_16",
            "time_sin",
            "base_load_lag_8",
            "base_load_delta_2",
            "base_load_lag_4",
            "time_cos",
            "timestep",
            "base_load_std_4",
            "base_load_std_8",
            "base_load_lag_12",
            "base_load_lag_2",
            "n_evs_at_home",
        ],
        "pv_gen": [
            "pv_gen",
            "pv_ma_2",
            "pv_lag_1",
            "pv_ma_4",
            "pv_lag_2",
            "pv_std_8",
            "pv_ma_8",
            "time_cos",
            "pv_delta_1",
            "pv_std_4",
            "pv_lag_4",
            "pv_delta_2",
            "steps_to_daylight_start",
            "steps_to_daylight_end",
            "timestep",
            "time_sin",
            "pv_accel",
            "pv_lag_8",
            "pv_lag_12",
            "pv_ma_16",
        ],
    },
    "ridge": {
        "ev_status": [
            "timestep",
            "steps_to_start1_earliest",
            "steps_to_end2_latest",
            "steps_to_start2_earliest",
            "steps_to_end1_latest",
            "start2_observed",
            "start2",
            "end2",
            "start1_observed",
            "observed_window_length_2",
            "end2_observed",
            "status",
            "window_length_slack_2",
            "end1_observed",
            "observed_window_length_1",
            "phase_id",
            "window_length_slack_1",
            "time_sin",
            "end1",
            "start1",
            "status_lag_1",
            "status_lag_2",
            "time_cos",
            "status_lag_4",
            "status_lag_8",
            "status_lag_8_is_pad",
            "steps_in_current_state",
        ],
        "base_load": [
            "base_load",
            "base_load_ma_2",
            "base_load_delta_1",
            "base_load_lag_1",
            "base_load_ma_4",
            "base_load_lag_2",
            "base_load_accel",
            "base_load_delta_2",
            "base_load_lag_8",
            "base_load_ma_16",
            "time_cos",
            "time_sin",
            "base_load_ma_8",
            "base_load_std_4",
            "timestep",
            "base_load_std_8",
            "base_load_lag_4",
            "n_evs_at_home",
            "base_load_lag_12",
        ],
        "pv_gen": [
            "pv_gen",
            "pv_ma_2",
            "pv_lag_1",
            "pv_lag_2",
            "pv_delta_1",
            "pv_ma_4",
            "pv_delta_2",
            "pv_std_8",
            "pv_lag_8",
            "pv_lag_4",
            "time_cos",
            "pv_std_4",
            "time_sin",
            "pv_accel",
            "pv_ma_16",
            "steps_to_daylight_start",
            "steps_to_daylight_end",
            "pv_lag_12",
            "timestep",
            "pv_ma_8",
        ],
    },
}


class ModelConfig:
    MODEL_TARGETS: tuple[str, ...] = MODEL_TARGETS
    MODEL_FEATURE_DOMAIN: dict[str, list[str]] = MODEL_FEATURE_DOMAIN
    MODEL_FEATURES_BY_FAMILY: dict[str, dict[str, list[str]]] = MODEL_FEATURES_BY_FAMILY

    @staticmethod
    def get_model_family_name(model: ModelLike) -> str:
        """Get family name from model instance.
            xgbclassifier
            xgbregressor
            randomforestclassifier
            randomforestregressor
            ridgeclassifier
            ridge
        """
        def _family_from_name(name: str) -> str:
            lower_name = name.lower()
            if "xgb" in lower_name or "xgboost" in lower_name:
                return "xgboost"
            if "randomforest" in lower_name:
                return "random_forest"
            if "ridge" in lower_name:
                return "ridge"
            return ""

        family = _family_from_name(model.__class__.__name__)
        if family:
            return family

        # Ridge models are often wrapped in sklearn.Pipeline; inspect known wrappers.
        named_steps = getattr(model, "named_steps", None)
        if isinstance(named_steps, dict):
            inner = named_steps.get("model")
            if inner is not None:
                family = _family_from_name(inner.__class__.__name__)
                if family:
                    return family

        for attr_name in ("estimator", "base_estimator", "final_estimator", "regressor", "classifier"):
            inner = getattr(model, attr_name, None)
            if inner is None:
                continue
            family = _family_from_name(inner.__class__.__name__)
            if family:
                return family

        raise ValueError(
            f"model family not found for class '{model.__class__.__name__}'. function needs fixing"
        )

    @staticmethod
    def _build_target_dirs(model_dir: Path) -> dict[str, Path]:
        return {target: Path(model_dir / target) for target in ModelConfig.MODEL_TARGETS}

    @staticmethod
    def build_model_family_configs(root_dir: Path) -> dict[str, "ModelFamilyConfig"]:
        xgb_dir = Path(root_dir / "training" / "xgboost" / "models")
        rf_dir = Path(root_dir / "training" / "random_forest" / "models")
        ridge_dir = Path(root_dir / "training" / "ridge" / "models")

        return {
            "xgb": ModelFamilyConfig(
                name="xgb",
                model_dir=xgb_dir,
                file_suffix=".json",
                target_model_dirs=ModelConfig._build_target_dirs(xgb_dir),
            ),
            "rf": ModelFamilyConfig(
                name="rf",
                model_dir=rf_dir,
                file_suffix=".pkl",
                target_model_dirs=ModelConfig._build_target_dirs(rf_dir),
            ),
            "ridge": ModelFamilyConfig(
                name="ridge",
                model_dir=ridge_dir,
                file_suffix=".pkl",
                target_model_dirs=ModelConfig._build_target_dirs(ridge_dir),
            ),
        }




def get_model_family_name(model: ModelLike)->str:
    """get family name from model instance.
    possible class names:
        xgbclassifier
        xgbregressor
        randomforestclassifier
        randomforestregressor
        ridgeclassifier
        ridge
    """

    return ModelConfig.get_model_family_name(model)



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
    return ModelConfig._build_target_dirs(model_dir)


def build_model_family_configs(root_dir: Path) -> dict[str, ModelFamilyConfig]:
    return ModelConfig.build_model_family_configs(root_dir)


if __name__ == "__main__":

    def dropped_features_report() -> None:
        for family in ModelConfig.MODEL_FEATURES_BY_FAMILY:
            for target in ModelConfig.MODEL_FEATURES_BY_FAMILY[family]:
                features = ModelConfig.MODEL_FEATURES_BY_FAMILY[family][target]
                domain_features = ModelConfig.MODEL_FEATURE_DOMAIN[target]
                missing_features = set(domain_features) - set(features)
                if missing_features:
                    print("we dropped these features for family", family, "target", target, ":", missing_features)

    dropped_features_report() # checks out

    '''
    we dropped these features for family XGBOOST target EV_STATUS : {'status_lag_4_is_pad', 'steps_to_end2_latest', 'max_commute_steps_2', 'end2_latest', 'start1_observed', 'start2_observed', 'status_lag_8_is_pad', 'observed_window_length_2', 'steps_to_end1_latest', 'end2', 'window_length_slack_2', 'status_lag_1_is_pad', 'start1_earliest', 'steps_to_start1_earliest', 'end2_observed', 'status_lag_2_is_pad', 'status_lag_8', 'start2_earliest', 'steps_to_start2_earliest', 'end1_observed', 'end1_latest', 'max_commute_steps_1'}
    we dropped these features for family XGBOOST target BASE_LOAD : {'base_load_lag_2_is_pad', 'base_load_lag_4_is_pad', 'base_load_lag_8_is_pad', 'base_load_lag_12_is_pad', 'base_load_lag_1_is_pad'}
    we dropped these features for family XGBOOST target PV_GEN : {'steps_to_daylight_end', 'pv_lag_8_is_pad', 'pv_lag_4_is_pad', 'pv_lag_12_is_pad', 'pv_lag_1_is_pad', 'pv_lag_2_is_pad', 'steps_to_daylight_start'}
    we dropped these features for family RANDOM_FOREST target EV_STATUS : {'status_lag_4_is_pad', 'status_lag_2_is_pad', 'start2_earliest', 'max_commute_steps_2', 'end2_latest', 'status_lag_1_is_pad', 'start1_earliest', 'end1_latest', 'status_lag_8_is_pad', 'max_commute_steps_1'}
    we dropped these features for family RANDOM_FOREST target BASE_LOAD : {'base_load_lag_2_is_pad', 'base_load_lag_4_is_pad', 'base_load_lag_8_is_pad', 'base_load_lag_12_is_pad', 'base_load_lag_1_is_pad'}
    we dropped these features for family RANDOM_FOREST target PV_GEN : {'pv_lag_8_is_pad', 'pv_lag_4_is_pad', 'pv_lag_12_is_pad', 'pv_lag_1_is_pad', 'pv_lag_2_is_pad'}
    we dropped these features for family RIDGE target EV_STATUS : {'status_lag_4_is_pad', 'status_lag_2_is_pad', 'start2_earliest', 'max_commute_steps_2','end2_latest', 'status_lag_1_is_pad', 'start1_earliest', 'end1_latest', 'max_commute_steps_1'}
    we dropped these features for family RIDGE target BASE_LOAD : {'base_load_lag_2_is_pad', 'base_load_lag_4_is_pad', 'base_load_lag_8_is_pad', 'base_load_lag_12_is_pad', 'base_load_lag_1_is_pad'}
    we dropped these features for family RIDGE target PV_GEN : {'pv_lag_8_is_pad', 'pv_lag_4_is_pad', 'pv_lag_12_is_pad', 'pv_lag_1_is_pad', 'pv_lag_2_is_pad'}
    '''