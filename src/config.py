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

    # Runtime simulation households (sorted by has_pv, has_bess).
    RUNTIME_SIM_PLAYER_IDS = [1, 27, 42, 68, 85, 92, 110, 114, 118, 167, 202, 229, 12, 52, 159, 215, 131, 153, 199, 238]

    # Five test folds (A-E), each holding four runtime households.
    RUNTIME_TEST_FOLDS = {
        "A": RUNTIME_SIM_PLAYER_IDS[:4],
        "B": RUNTIME_SIM_PLAYER_IDS[4:8],
        "C": RUNTIME_SIM_PLAYER_IDS[8:12],
        "D": RUNTIME_SIM_PLAYER_IDS[12:16],
        "E": RUNTIME_SIM_PLAYER_IDS[16:20],
    }

    # Reverse lookup used by predictors: household -> fold id.
    RUNTIME_PLAYER_TO_TEST_FOLD = {
        _player_id: _fold_id
        for _fold_id, _player_ids in RUNTIME_TEST_FOLDS.items()
        for _player_id in _player_ids
    }

    PLAYERS_WITH_BESS = [1, 4, 6, 8, 9, 10, 13, 16, 17, 18, 19, 21, 22, 23, 24, 25, 27, 28, 29, 30, 34, 35, 37, 38, 40, 41, 42, 44, 45, 53, 54, 56, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 70, 71, 73, 75, 77, 78, 79, 81, 82, 85, 86, 87, 88, 89, 90, 91, 92, 93, 96, 97, 98, 99, 100, 101, 102,105, 106, 107, 108, 109, 110, 111, 114, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 129, 130, 132, 135, 136, 137, 138, 139, 140, 143, 144, 147, 149, 152, 154, 155, 157, 158, 161, 162, 163, 165, 166, 167, 168, 169, 170, 172, 175, 176, 179, 180, 182, 183, 185, 186, 189, 190, 192, 194, 197, 200, 201, 202, 203, 206, 210, 213, 216, 217, 220, 221, 222, 224, 227, 228, 229, 230, 232, 235, 236, 239, 241, 242, 243]
    PLAYERS_WITH_PV = [1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 34, 35, 37, 38, 40, 41, 42, 44, 45, 47, 48, 49, 50, 52, 53, 54, 55, 56, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 81, 82, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 105, 106, 107, 108, 109, 110, 111, 114, 115, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 129, 130, 132, 134, 135, 136, 137, 138, 139, 140, 143, 144, 145, 147, 148, 149, 151, 152, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 179, 180, 181, 182, 183, 184, 185, 186, 189, 190, 192, 193, 194, 195, 197, 198, 200, 201, 202, 203, 205, 206, 207, 209, 210, 211, 213, 215, 216, 217, 218, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 234, 235, 236, 239, 240, 241, 242, 243, 246, 247, 250]

    @classmethod
    def get_test_fold_for_player(cls, player_id: int) -> str:
        try:
            return cls.RUNTIME_PLAYER_TO_TEST_FOLD[int(player_id)]
        except KeyError as exc:
            raise KeyError(f"Player ID {player_id} is not part of the runtime fold set.") from exc

    @classmethod
    def get_model_path(cls, family: str, target: str, player_id: int) -> Path:
        family_key = str(family).lower()
        if family_key not in cls.MODEL_FAMILY_CONFIGS:
            valid_families = ", ".join(sorted(cls.MODEL_FAMILY_CONFIGS.keys()))
            raise ValueError(
                f"Unsupported model family '{family_key}'. Expected one of: {valid_families}"
            )

        fold_id = cls.get_test_fold_for_player(int(player_id))
        family_config = cls.MODEL_FAMILY_CONFIGS[family_key]
        return family_config.get_model_path_for_fold(target=target, fold_id=fold_id)

    @classmethod
    def get_xgb_model_path(cls, target: str, player_id: int) -> Path:
        return cls.get_model_path(family="xgb", target=target, player_id=player_id)

    @classmethod
    def get_rf_model_path(cls, target: str, player_id: int) -> Path:
        return cls.get_model_path(family="rf", target=target, player_id=player_id)

    @classmethod
    def get_ridge_model_path(cls, target: str, player_id: int) -> Path:
        return cls.get_model_path(family="ridge", target=target, player_id=player_id)


    XGB_FEATURES = MODEL_FEATURE_DOMAIN
    MODEL_FEATURES = MODEL_FEATURE_DOMAIN