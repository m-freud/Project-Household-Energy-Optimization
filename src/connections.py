import sqlite3
from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

from src.config import Config


def create_influx_client() -> InfluxDBClient:
    return InfluxDBClient(url=Config.INFLUX_URL, token=Config.INFLUX_TOKEN, org=Config.INFLUX_ORG)


def get_influx_buckets_api():
    return create_influx_client().buckets_api()


def get_influx_write_api():
    return create_influx_client().write_api(SYNCHRONOUS)


def get_influx_query_api():
    return create_influx_client().query_api()


def create_sqlite_connection():
    return sqlite3.connect(Config.SQLITE_PATH)


def fetch_multiple_timeseries(influx_query_api, player_id, measurements:list):
    timeseries_data = {m: [] for m in measurements}

    flux_set = "[" + ",".join(f'"{m}"' for m in measurements) + "]"

    
    query = f'''
    from(bucket: "{Config.INFLUX_BUCKET}")
        |> range(start: 0)
        |> filter(fn: (r) => r["player_id"] == "{player_id}")

        |> filter(fn: (r) => contains(value: r["_measurement"], set: {flux_set}))
        |> sort(columns: ["_time"])
    '''
    result = influx_query_api.query(org=Config.INFLUX_ORG, query=query)


    for table in result:
        measurement = table.records[0].values['_measurement']
        series = [record.get_value() for record in table]
        timeseries_data[measurement] = series
    
    return timeseries_data

