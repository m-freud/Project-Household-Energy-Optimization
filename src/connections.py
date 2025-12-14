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
INFLUX_BUCKET = Config.INFLUX_BUCKET
INFLUX_ORG = Config.INFLUX_ORG
assert INFLUX_URL and INFLUX_TOKEN and INFLUX_ORG and INFLUX_BUCKET

influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)

# SQLite connection
SQL_PATH = ROOT_DIR / "sqlite" / "energy.db"

sqlite_conn = sqlite3.connect(SQL_PATH)

def get_influx_client() -> InfluxDBClient:
    return influx_client

def get_sqlite_connection() -> sqlite3.Connection:
    return sqlite_conn

def close_connections():
    influx_client.close()
    sqlite_conn.close()
