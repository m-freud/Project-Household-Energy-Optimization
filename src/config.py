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