from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

class Config:
    ROOT_DIR = Path(__file__).parent.parent
    XGB_MODEL_DIR = Path(ROOT_DIR / "training" / "xgboost" / "models")

    XGB_BASE_LOAD_MODEL_PATH = Path(XGB_MODEL_DIR / "base_load_regressor.json")
    XGB_PV_GEN_MODEL_PATH = Path(XGB_MODEL_DIR / "pv_gen_regressor.json")
    XGB_EV_STATUS_MODEL_PATH = Path(XGB_MODEL_DIR / "ev_status_classifier.json")

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

    HOUSEHOLD_SET_NO_DUPLICATES = [1, 2, 6, 12, 45, 56, 58, 63, 80, 85, 92, 95, 102, 103, 108, 131, 141, 145, 146, 150, 152, 153, 170, 178, 192, 199, 202, 203, 204, 206, 207, 208, 210, 212, 214, 220, 231, 242, 244, 247]
    H_SET_TESTING = HOUSEHOLD_SET_NO_DUPLICATES[:int(len(HOUSEHOLD_SET_NO_DUPLICATES)*0.2)]
    H_SET_TRAINING = HOUSEHOLD_SET_NO_DUPLICATES[int(len(HOUSEHOLD_SET_NO_DUPLICATES)*0.2):]

    CLEAN_SET = [1, 12, 27, 42, 52, 68, 85, 92, 110, 114, 118, 131, 153, 159, 167, 199, 202, 215, 229, 238]

    CLEAN_SUBSETS = {
        "A": CLEAN_SET[:4],
        "B": CLEAN_SET[4:8],
        "C": CLEAN_SET[8:12],
        "D": CLEAN_SET[12:16],
        "E": CLEAN_SET[16:20],
    }

    CLEAN_COMPLEMENTS = {
        "A": CLEAN_SET[4:],
        "B": CLEAN_SET[:4] + CLEAN_SET[8:],
        "C": CLEAN_SET[:8] + CLEAN_SET[12:],
        "D": CLEAN_SET[:12] + CLEAN_SET[16:],
        "E": CLEAN_SET[:16],
    }

    EXTRA_IDS = {
        "base_load": [],
        "pv_gen": [4, 6, 9, 11, 14, 18, 23, 30, 34, 37, 40, 45, 49, 56, 58, 59, 63, 66, 67, 69, 70, 71, 77, 78, 79, 93, 102, 152, 162, 206, 207, 209, 210, 211, 216, 220],
        "ev1_status": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 90, 91, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 108, 109, 111, 112, 113, 115, 116, 117, 119, 120, 121, 122, 123, 125, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 150, 151, 152, 154, 155, 156, 157, 158, 160, 161, 162, 163, 164, 165, 166, 168, 169, 170, 171, 173, 174, 175, 176, 177, 178, 179, 180, 181, 183, 184, 185, 186, 187, 188, 190, 191, 192, 193, 194, 195, 196, 197, 198, 200, 201, 203, 204, 206, 207, 209, 210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 230, 231, 232, 234, 235, 237, 239, 240, 241, 242, 243, 244, 245, 246, 248, 250],
        "ev2_status":[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 88, 89, 90, 91, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 105, 106, 107, 108, 109, 111, 112, 115, 116, 117, 120, 121, 122, 123, 125, 126, 129, 130, 132, 134, 135, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 155, 156, 157, 158, 160, 161, 162, 163, 164, 166, 168, 170, 171, 172, 173, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 186, 188, 189, 190, 191, 192, 193, 194, 196, 197, 200, 201, 203, 204, 205, 206, 207, 208, 209, 210, 212, 213, 214, 216, 217, 219, 220, 221, 222, 223, 224,225, 226, 227, 228, 230, 231, 232, 233, 234, 235, 236, 237, 239, 240, 241, 243, 244, 246, 248, 249, 250],
    }

    TRAIN_SETS = {
        label: {
            metric: sorted(set(CLEAN_COMPLEMENTS[label] + extra_ids))
            for metric, extra_ids in EXTRA_IDS.items()
        }
        for label in CLEAN_COMPLEMENTS
    }

    XGB_FEATURES = {
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
        ]
    }