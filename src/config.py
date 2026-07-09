from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

class Config:
    ROOT_DIR = Path(__file__).parent.parent

    # data source
    EXCEL_FILE_PATH = ROOT_DIR / "data" / "A.xlsx"

    # sqlite
    SQLITE_PATH = Path(os.getenv("SQLITE_PATH", str(ROOT_DIR / "sqlite" / "energy.db")))

    DURATION_TIMESTEP = 0.25 # 4 per hour

    # Expected EV charging unavailability windows (global user-style settings).
    # Values are hardcoded from research/ev_unavailable_windows.csv and represent
    # approximate commute periods, not realized household-specific future states.
    EV_UNAVAILABLE_WINDOWS_OBSERVED = {
        "ev1": [
            {"window": 1, "earliest_start": 33, "latest_end": 48, "max_unavailable_steps": 5},
            {"window": 2, "earliest_start": 71, "latest_end": 85, "max_unavailable_steps": 5},
        ],
        "ev2": [
            {"window": 1, "earliest_start": 36, "latest_end": 49, "max_unavailable_steps": 4},
            {"window": 2, "earliest_start": 76, "latest_end": 87, "max_unavailable_steps": 4},
        ],
    }

    EV_UNAVAILABLE_WINDOWS_ALLOWED = {
        "ev1": [
            {"window": 1, "earliest_start": 32, "latest_end": 50, "max_unavailable_steps": 5},
            {"window": 2, "earliest_start": 70, "latest_end": 88, "max_unavailable_steps": 5},
        ],
        "ev2": [ #identical to ev1
            {"window": 1, "earliest_start": 32, "latest_end": 50, "max_unavailable_steps": 5},
            {"window": 2, "earliest_start": 70, "latest_end": 88, "max_unavailable_steps": 5},
        ],
    }   

    # PV generation availability window (quarter-hour periods, 1..96).
    # Derived from the current dataset's all-household average PV curve.
    PV_GENERATION_WINDOW_OBSERVED = {
        "earliest_start": 32,
        "latest_end": 71,
    }
