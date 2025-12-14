from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

class Config:
    ROOT_DIR = Path(__file__).parent.parent

    SQL_PATH = ROOT_DIR / "sqlite" / "energy.db"

    INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "")
    INFLUX_URL = os.getenv("INFLUX_URL", "")
    INFLUX_ORG = os.getenv("INFLUX_ORG", "")
    INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "")
