from dotenv import load_dotenv
import os
from pathlib import Path

from src.simulation.controllers.mpc.predictors.ml.model_config import (
    MODEL_FEATURE_DOMAIN,
    build_model_family_configs,
)

# warning circular import
# from training.split.clean_split import PARTITIONS

load_dotenv()

class RuntimeConfig: # TODO remove ABCDE fold logic
    ROOT_DIR = Path(__file__).parent.parent
    MODEL_FAMILY_CONFIGS = build_model_family_configs(ROOT_DIR)

    # Backward-compatible aliases.
    XGB_MODEL_DIR = MODEL_FAMILY_CONFIGS["xgb"].model_dir
    RF_MODEL_DIR = MODEL_FAMILY_CONFIGS["rf"].model_dir
    RIDGE_MODEL_DIR = MODEL_FAMILY_CONFIGS["ridge"].model_dir

    XGB_METRIC_MODEL_DIRS = MODEL_FAMILY_CONFIGS["xgb"].target_model_dirs
    RF_METRIC_MODEL_DIRS = MODEL_FAMILY_CONFIGS["rf"].target_model_dirs
    RIDGE_METRIC_MODEL_DIRS = MODEL_FAMILY_CONFIGS["ridge"].target_model_dirs

    # data source
    EXCEL_FILE_PATH = Path(os.getenv("EXCEL_FILE_PATH", str(ROOT_DIR / "data" / "energy_community_data.xlsx")))

    # sqlite
    SQLITE_PATH = Path(os.getenv("SQLITE_PATH", str(ROOT_DIR / "sqlite" / "energy.db")))

    DURATION_TIMESTEP = 0.25 # 4 per hour
    TOTAL_TIMESTEPS_DAY = int(24 / DURATION_TIMESTEP)

    # observed ev availabilities
    EV_COMMUTE_WINDOWS_OBSERVED = {
        "ev1": [
            {"window": 1, "earliest_start": 33, "latest_end": 48, "max_unavailable_steps": 5},
            {"window": 2, "earliest_start": 71, "latest_end": 85, "max_unavailable_steps": 5},
        ],
        "ev2": [
            {"window": 1, "earliest_start": 36, "latest_end": 49, "max_unavailable_steps": 4},
            {"window": 2, "earliest_start": 76, "latest_end": 87, "max_unavailable_steps": 4},
        ],
    }

    # rounded allowed windows for predictions
    EV_COMMUTE_WINDOWS_ALLOWED = {
        "ev1": [
            {"window": 1, "earliest_start": 32, "latest_end": 50, "max_unavailable_steps": 5},
            {"window": 2, "earliest_start": 70, "latest_end": 88, "max_unavailable_steps": 5},
        ],
        "ev2": [ #identical to ev1
            {"window": 1, "earliest_start": 32, "latest_end": 50, "max_unavailable_steps": 5},
            {"window": 2, "earliest_start": 70, "latest_end": 88, "max_unavailable_steps": 5},
        ],
    }   

    # pv window for predictions
    PV_GENERATION_WINDOW_OBSERVED = {
        "earliest_start": 32,
        "latest_end": 71,
    }

    PV_GENERATION_WINDOW_ALLOWED = PV_GENERATION_WINDOW_OBSERVED

    PLAYERS_WITH_PV = [1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 34, 35, 37, 38, 40, 41, 42, 44, 45, 47, 48, 49, 50, 52, 53, 54, 55, 56, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 81, 82, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 105, 106, 107, 108, 109, 110, 111, 114, 115, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 129, 130, 132, 134, 135, 136, 137, 138, 139, 140, 143, 144, 145, 147, 148, 149, 151, 152, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 179, 180, 181, 182, 183, 184, 185, 186, 189, 190, 192, 193, 194, 195, 197, 198, 200, 201, 202, 203, 205, 206, 207, 209, 210, 211, 213, 215, 216, 217, 218, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 234, 235, 236, 239, 240, 241, 242, 243, 246, 247, 250]

    XGB_FEATURES = MODEL_FEATURE_DOMAIN
    MODEL_FEATURES = MODEL_FEATURE_DOMAIN

    ALL_PLAYER_IDS = tuple(range(1, 251))

    # minimal set that covers all base load groups while no households share any profile shapes
    # used for quick testing of the simulation and ML models
    INDEPENDENT_TEST_SET_20 = [1, 12, 27, 42, 52, 68, 85, 92, 110, 114, 118, 131, 153, 159, 167, 199, 202, 215, 229, 238]

    # PARTITIONS = PARTITIONS


    def get_fold_id(self, target: str, household_id: int):
        return self.get_fold_for_player(target=target, player_id=household_id)

    @classmethod
    def get_fold_for_player(cls, target: str, player_id: int) -> str:
        try:
            return cls.FOLD_PLAYER_TO_FOLD_BY_TARGET[str(target)][int(player_id)]
        except KeyError as exc:
            raise ValueError(
                f"Household ID {player_id} not found in any fold for target {target}."
            ) from exc

    @classmethod
    def get_fold_members(cls, target: str, fold_id: str) -> list[int]:
        try:
            return list(cls.FOLD_SPLITS_BY_TARGET[str(target)][str(fold_id)])
        except KeyError as exc:
            raise ValueError(f"Unknown fold '{fold_id}' for target '{target}'.") from exc

    @classmethod
    def get_training_ids_for_fold(cls, target: str, fold_id: str) -> list[int]:
        test_ids = set(cls.get_fold_members(target=target, fold_id=fold_id))
        target_population_ids = cls.get_target_population_ids(target)
        return [player_id for player_id in target_population_ids if player_id not in test_ids]

    @classmethod
    def get_target_population_ids(cls, target: str) -> list[int]:
        target_key = str(target)
        try:
            fold_splits = cls.FOLD_SPLITS_BY_TARGET[target_key]
        except KeyError as exc:
            raise ValueError(f"Unknown target '{target}'.") from exc

        population_ids = {
            int(player_id)
            for fold_members in fold_splits.values()
            for player_id in fold_members
        }
        return sorted(population_ids)

    @classmethod
    def get_player_to_fold_map(cls, target: str) -> dict[int, str]:
        try:
            return dict(cls.FOLD_PLAYER_TO_FOLD_BY_TARGET[str(target)])
        except KeyError as exc:
            raise ValueError(f"Unknown target '{target}'.") from exc

    @classmethod
    def get_model_path_for_fold(cls, family: str, target: str, fold_id: str) -> Path:
        family_key = str(family).lower()
        if family_key not in cls.MODEL_FAMILY_CONFIGS:
            valid_families = ", ".join(sorted(cls.MODEL_FAMILY_CONFIGS.keys()))
            raise ValueError(
                f"Unsupported model family '{family_key}'. Expected one of: {valid_families}"
            )

        family_config = cls.MODEL_FAMILY_CONFIGS[family_key]
        return family_config.get_model_path_for_fold(target=target, fold_id=fold_id)

    @classmethod
    def get_model_path(cls, family: str, target: str, player_id: int) -> Path:
        fold_id = cls.get_fold_for_player(target=target, player_id=player_id)
        return cls.get_model_path_for_fold(family=family, target=target, fold_id=fold_id)

