import sqlite3
from influxdb_client.client.influxdb_client import InfluxDBClient

from src.config import Config


def create_influx_client() -> InfluxDBClient:
    return InfluxDBClient(url=Config.INFLUX_URL, token=Config.INFLUX_TOKEN, org=Config.INFLUX_ORG)


def get_influx_buckets_api():
    return create_influx_client().buckets_api()


def get_influx_write_api():
    return create_influx_client().write_api()


def get_influx_query_api():
    return create_influx_client().query_api()


def create_sqlite_connection():
    return sqlite3.connect(Config.SQLITE_PATH)
