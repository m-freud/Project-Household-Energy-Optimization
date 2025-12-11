import sys
import sqlite3
from pathlib import Path
from influxdb_client.client.influxdb_client import InfluxDBClient
import pandas as pd


ROOT_DIR = Path(__file__).parent.parent.parent.parent
SQL_PATH = ROOT_DIR / "sqlite" / "energy.db"

# config file import
sys.path.append(str(ROOT_DIR))
from config import Config

INFLUX_BUCKET = Config.INFLUX_BUCKET
INFLUX_ORG = Config.INFLUX_ORG
assert INFLUX_BUCKET and INFLUX_ORG


class Household:
    def __init__(self, influx_client, sql_conn, player_id):
        self.influx_client = influx_client
        self.influx_query_api = influx_client.query_api()

        self.sql_conn = sql_conn
        self.sql_cursor = sql_conn.cursor()

        self.player_id = player_id

        self.profiles = {}

        for profile in ["load", "pv", "buy_price", "sell_price"]:
            self.profiles[profile] = self.fetch_timeseries(measurement=profile, player_id=player_id)

        self.has_ess, self.has_pv = self.get_player_info(table="player_pv_ess", fields=["has_ess", "has_pv"])

        if self.has_ess:
            self.ess_model_id = self.get_player_info(table="bess", fields="model_id") if self.has_ess else None


        print(f"Initialized Household for player_id {player_id} | has_pv: {self.has_pv} | has_ess: {self.has_ess} | ess_model_id: {self.ess_model_id}")


    def fetch_timeseries(self, measurement: str, player_id: int):
        query = f'''
        from(bucket: "{INFLUX_BUCKET}")
            |> range(start: 0)
            |> filter(fn: (r) => r._measurement == "{measurement}" and r.player_id == "{player_id}")
            '''
        
        query_result = self.influx_query_api.query(org=INFLUX_ORG, query=query)

        results = []
        for table in query_result:
            for record in table.records:
                results.append((record.get_value()))

        return results
    

    def get_player_info(self, table, fields: list[str] | str):
        # Normalize fields to a list
        if isinstance(fields, str):
            fields = [fields]

        results = []
        for field in fields:
            # use parameterized query to avoid SQL injection and fetch a single row
            self.sql_cursor.execute(f"SELECT {field} FROM {table} WHERE player_id = {self.player_id}")
            row = self.sql_cursor.fetchone()
            results.append(row[0] if row and len(row) > 0 else None)

        # return a single value when one field was requested, otherwise a tuple for unpacking
        if len(results) == 1:
            return results[0]
        return tuple(results)
    

    def generate_cost_profile(self, load_profile):
        buy_price = self.profiles["buy_price"]
        sell_price = self.profiles["sell_price"]

        if "bess" in self.profiles:
            bess_profile = self.profiles["bess"]

        cost_profile = []
        for load, buy_price, sell_price in zip(load_profile, buy_price, sell_price):
            if load >= 0:
                cost_profile.append(load * buy_price)
            else:
                cost_profile.append(load * sell_price * -1)
        return cost_profile
    

if __name__ == "__main__":
    # sqlite connection
    conn = sqlite3.connect(SQL_PATH)

    # Influx connection
    INFLUX_TOKEN = Config.INFLUX_TOKEN
    INFLUX_URL = Config.INFLUX_URL
    INFLUX_ORG = Config.INFLUX_ORG
    INFLUX_BUCKET = Config.INFLUX_BUCKET
    assert INFLUX_URL and INFLUX_TOKEN and INFLUX_ORG and INFLUX_BUCKET

    influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)

    household_1 = Household(influx_client=influx_client, sql_conn=conn, player_id=1)

    # quick plot for all profiles
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(household_1.profiles["load"], label="Load Profile")
    plt.plot(household_1.profiles["pv"], label="PV Profile")
    plt.plot(household_1.profiles["buy_price"], label="Buy Price Profile")
    plt.plot(household_1.profiles["sell_price"], label="Sell Price Profile")
    plt.legend()
    plt.show()