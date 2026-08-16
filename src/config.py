from dotenv import load_dotenv
import os
from pathlib import Path

from src.simulation.controllers.mpc.predictors.ml.model_config import (
    MODEL_FEATURE_DOMAIN,
    build_model_family_configs,
)

load_dotenv()

class Config:
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

    PLAYERS_WITH_BESS = [1, 4, 6, 8, 9, 10, 13, 16, 17, 18, 19, 21, 22, 23, 24, 25, 27, 28, 29, 30, 34, 35, 37, 38, 40, 41, 42, 44, 45, 53, 54, 56, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 70, 71, 73, 75, 77, 78, 79, 81, 82, 85, 86, 87, 88, 89, 90, 91, 92, 93, 96, 97, 98, 99, 100, 101, 102,105, 106, 107, 108, 109, 110, 111, 114, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 129, 130, 132, 135, 136, 137, 138, 139, 140, 143, 144, 147, 149, 152, 154, 155, 157, 158, 161, 162, 163, 165, 166, 167, 168, 169, 170, 172, 175, 176, 179, 180, 182, 183, 185, 186, 189, 190, 192, 194, 197, 200, 201, 202, 203, 206, 210, 213, 216, 217, 220, 221, 222, 224, 227, 228, 229, 230, 232, 235, 236, 239, 241, 242, 243]
    PLAYERS_WITH_PV = [1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 34, 35, 37, 38, 40, 41, 42, 44, 45, 47, 48, 49, 50, 52, 53, 54, 55, 56, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 81, 82, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 105, 106, 107, 108, 109, 110, 111, 114, 115, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 129, 130, 132, 134, 135, 136, 137, 138, 139, 140, 143, 144, 145, 147, 148, 149, 151, 152, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 179, 180, 181, 182, 183, 184, 185, 186, 189, 190, 192, 193, 194, 195, 197, 198, 200, 201, 202, 203, 205, 206, 207, 209, 210, 211, 213, 215, 216, 217, 218, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 234, 235, 236, 239, 240, 241, 242, 243, 246, 247, 250]

    XGB_FEATURES = MODEL_FEATURE_DOMAIN
    MODEL_FEATURES = MODEL_FEATURE_DOMAIN

    ALL_PLAYER_IDS = tuple(range(1, 251))

    # minimal set that covers all base load groups while no households share any profile shapes
    # used for quick testing of the simulation and ML models
    INDEPENDENT_TEST_SET = [1, 12, 27, 42, 52, 68, 85, 92, 110, 114, 118, 131, 153, 159, 167, 199, 202, 215, 229, 238]

    FOLD_IDS = ("fold_1", "fold_2", "fold_3", "fold_4", "fold_5")

    TARGET_FOLD_SPLITS = {
        "base_load": {
            'fold_1': [1, 5, 15, 18, 23, 25, 26, 30, 33, 34, 37, 41, 45, 46, 49, 50, 106, 107, 109, 110, 111, 116, 120, 122, 128, 129, 132, 140, 142, 143, 147, 148, 8, 13, 17, 19, 21, 24, 27, 31, 35, 36, 38, 39, 44, 2, 3, 4, 12, 14],
            'fold_2': [6, 7, 9, 10, 11, 16, 20, 22, 28, 29, 32, 40, 42, 43, 47, 48, 151, 155, 165, 168, 173, 175, 176, 180, 183, 184, 187, 191, 195, 196, 199, 200, 58, 63, 67, 69, 71, 74, 77, 81, 85, 86, 88, 89, 94, 52, 53, 54, 62, 64],
            'fold_3': [51, 55, 65, 68, 73, 75, 76, 80, 83, 84, 87, 91, 95, 96, 99, 100, 156, 157, 159, 160, 161, 166, 170, 172, 178, 179, 182, 190, 192, 193, 197, 198, 108, 113, 117, 119, 121, 124, 127, 131, 135, 136, 138, 139, 144, 102, 103, 104, 112, 114],
            'fold_4': [56, 57, 59, 60, 61, 66, 70, 72, 78, 79, 82, 90, 92, 93, 97, 98, 201, 205, 215, 218, 223, 225, 226, 230, 233, 234, 237, 241, 245, 246, 249, 250, 158, 163, 167, 169, 171, 174, 177, 181, 185, 186, 188, 189, 194, 152, 153, 154, 162, 164],
            'fold_5': [101, 105, 115, 118, 123, 125, 126, 130, 133, 134, 137, 141, 145, 146, 149, 150, 206, 207, 209, 210, 211, 216, 220, 222, 228, 229, 232, 240, 242, 243, 247, 248, 208, 213, 217, 219, 221, 224, 227, 231, 235, 236, 238, 239, 244, 202, 203, 204, 212, 214]
            },
        "pv_gen": {
            'fold_1': [7, 48, 52, 115, 130, 137, 163, 166, 174, 181, 182, 218, 21, 24, 28, 73, 126, 157, 175, 203, 215, 235, 250, 32, 106, 119, 155, 172, 183, 189, 193, 202, 246, 85, 88, 89, 45, 50, 71, 86, 216, 232, 23, 63, 93, 207],
            'fold_2': [9, 13, 65, 109, 132, 145, 148, 180, 186, 213, 223, 31, 68, 101, 117, 143, 144, 165, 185, 194, 200, 217, 110, 136, 173, 179, 190, 197, 221, 226, 231, 234, 209, 242, 247, 56, 60, 77, 94, 229, 243, 30, 66, 102, 211],
            'fold_3': [10, 44, 62, 64, 96, 127, 159, 160, 169, 170, 227, 38, 47, 87, 118, 125, 129, 158, 184, 230, 236, 239, 6, 8, 53, 116, 124, 156, 220, 228, 240, 58, 81, 78, 98, 4, 34, 69, 114], 
            'fold_4': [11, 35, 54, 76, 100, 105, 107, 138, 161, 171, 176, 17, 19, 42, 108, 139, 168, 192, 195, 225, 241, 40,147, 149, 151, 177, 1, 41, 67, 74, 152, 154, 12, 37, 79, 162],
            'fold_5': [16, 20, 22, 27, 55, 75, 123, 134, 140, 205, 224, 29, 91, 99, 111, 120, 122, 135, 167, 198, 201, 59, 82, 90, 18, 25, 70, 97, 210, 222, 14, 49, 92, 206]
            },
        "ev1_status": {
            'fold_1': [7, 107, 247, 47, 124, 104, 189, 2, 9, 14, 21, 26, 31, 37, 42, 48, 57, 62, 67, 72, 77, 82, 88, 94, 99, 106, 112, 117, 122, 128,133, 138, 143, 150, 155, 160, 165, 171, 177, 183, 188, 194, 199, 204, 211, 216, 221, 227, 232, 239, 244],  
            'fold_2': [5, 49, 51, 172, 105, 149, 3, 10, 15, 22, 27, 33, 38, 43, 50, 58, 63, 68, 73, 78, 83, 90, 95, 100, 108, 113, 118, 123, 129, 134, 139, 145, 151, 156, 161, 166, 173, 178, 184, 190, 195, 200, 206, 212, 217, 223, 228, 234, 240, 245],  
            'fold_3': [17, 167, 52, 205, 144, 182, 4, 11, 16, 23, 28, 34, 39, 44, 53, 59, 64, 69, 74, 79, 84, 91, 96, 101, 109, 114, 119, 125, 130, 135, 140, 146, 152, 157, 162, 168, 174, 179, 185, 191, 196, 201, 207, 213, 218, 224, 229, 235, 241, 246],  
            'fold_4': [18, 236,54, 89, 222, 249, 6, 12, 19, 24, 29, 35, 40, 45, 55, 60, 65, 70, 75, 80, 85, 92, 97, 102, 110, 115, 120, 126, 131, 136, 141, 147, 153, 158, 163, 169, 175, 180, 186, 192, 197, 202, 209, 214, 219, 225, 230, 237, 242, 248],  
            'fold_5': [32, 233, 87, 208, 1, 8, 13, 20, 25, 30, 36, 41, 46, 56, 61, 66, 71, 76,81, 86, 93, 98, 103, 111, 116, 121, 127, 132, 137, 142, 148, 154, 159, 164, 170, 176, 181, 187, 193, 198, 203, 210, 215, 220, 226, 231, 238, 243, 250]
            },
        "ev2_status": {
            'fold_1': [4, 44, 26, 247, 66, 119, 102, 137, 130, 143, 156, 211, 5, 11, 19, 24, 30, 35, 42, 48, 53, 58, 65, 73, 78, 85, 92, 97, 103, 111, 118, 126, 140, 146, 153, 161, 167, 173, 179, 185, 191, 197, 203, 208, 214, 220, 225, 230, 235, 240, 248],  
            'fold_2': [9, 67, 36, 113, 71, 110, 105, 128, 131, 198, 160, 187, 6, 13, 20, 25, 31, 37, 43, 49, 54, 59, 68, 74, 79, 86, 93, 98, 106, 112, 121, 132, 141, 148, 155, 162, 168, 175, 180, 186, 192, 199, 204, 209, 215, 221, 226, 231, 236, 241, 249],  
            'fold_3': [12, 104, 41, 154, 80, 127, 114, 169, 138, 218, 1, 7, 14, 21, 27, 32, 38, 45, 50, 55, 61, 69,75, 81, 88, 94, 99, 107, 115, 122, 134, 142, 149, 157, 163, 170, 176, 182, 188, 193, 200, 205, 210, 216, 222, 227, 232, 237, 243, 250],  
            'fold_4': [15, 195, 60, 165, 82, 124, 120, 181, 147, 174, 2, 8, 17, 22, 28, 33, 39, 46, 51, 56, 63, 70, 76, 83, 90, 95, 100, 108, 116, 123, 135, 144, 151, 158, 164, 171, 177, 183, 189, 194, 201, 206, 212, 217, 223, 228, 233, 238, 244],  
            'fold_5': [16, 245, 62, 87, 89, 133, 129, 136, 150, 242, 3, 10, 18, 23, 29, 34, 40,47, 52, 57, 64, 72, 77, 84, 91, 96, 101, 109, 117, 125, 139, 145, 152, 159, 166, 172, 178, 184, 190, 196, 202, 207, 213, 219, 224, 229, 234, 239, 246]
            },
    }

    FOLD_SPLITS_BY_TARGET = TARGET_FOLD_SPLITS
    FOLD_PLAYER_TO_FOLD_BY_TARGET = {
        target: {
            household_id: fold_id
            for fold_id, household_ids in fold_splits.items()
            for household_id in household_ids
        }
        for target, fold_splits in TARGET_FOLD_SPLITS.items()
    }

    # Backward-compatible aliases for older runtime code.
    RUNTIME_TEST_FOLDS = FOLD_SPLITS_BY_TARGET["base_load"]
    RUNTIME_PLAYER_TO_TEST_FOLD = FOLD_PLAYER_TO_FOLD_BY_TARGET["base_load"]

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

