import sys
import sqlite3
from pathlib import Path
from influxdb_client.client.influxdb_client import InfluxDBClient

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
from config import Config

# InfluxDB connection
INFLUX_URL = Config.INFLUX_URL
INFLUX_TOKEN = Config.INFLUX_TOKEN
INFLUX_ORG = Config.INFLUX_ORG

# SQLite connection
SQL_PATH = ROOT_DIR / "sqlite" / "energy.db"

def create_influx_client() -> InfluxDBClient:
    assert INFLUX_URL and INFLUX_TOKEN and INFLUX_ORG
    return InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)


def create_sqlite_connection():
    return sqlite3.connect(SQL_PATH)
