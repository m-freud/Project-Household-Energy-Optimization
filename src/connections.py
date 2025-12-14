import sqlite3
from influxdb_client.client.influxdb_client import InfluxDBClient

from config import Config


def create_influx_client() -> InfluxDBClient:
    return InfluxDBClient(url=Config.INFLUX_URL, token=Config.INFLUX_TOKEN, org=Config.INFLUX_ORG)


def create_sqlite_connection():
    return sqlite3.connect(Config.SQL_PATH)
