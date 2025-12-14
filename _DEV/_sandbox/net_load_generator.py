import sys
from pathlib import Path
import sqlite3
from influxdb_client.client.influxdb_client import InfluxDBClient
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent
DB_PATH = ROOT_DIR / "sqlite" / "energy.db"

# config file import
sys.path.append(str(ROOT_DIR))
from config import Config

# sqlite connection
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Influx connection
INFLUX_TOKEN = Config.INFLUX_TOKEN
INFLUX_URL = Config.INFLUX_URL
INFLUX_ORG = Config.INFLUX_ORG
INFLUX_BUCKET = Config.INFLUX_BUCKET
assert INFLUX_URL and INFLUX_TOKEN and INFLUX_ORG and INFLUX_BUCKET

client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()


def fetch_timeseries(measurement: str, player_id: int):
    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
        |> range(start: 0)
        |> filter(fn: (r) => r._measurement == "{measurement}" and r.player_id == "{player_id}")
        '''
    
    query_result = query_api.query(org=INFLUX_ORG, query=query)

    results = []
    for table in query_result:
        for record in table.records:
            results.append((record.get_value()))

    return results


def generate_net_load(player_id: int):
    load = fetch_timeseries(measurement="load", player_id=player_id)
    pv = fetch_timeseries(measurement="pv", player_id=player_id)

    # return difference of load and pv
    net_load = [l - p for l, p in zip(load, pv)]

    return net_load


def generate_net_cost(player_id: int):
    net_load = generate_net_load(player_id=player_id)
    buy_price = fetch_timeseries(measurement="buy_price", player_id=player_id)
    sell_price = fetch_timeseries(measurement="sell_price", player_id=player_id)

    net_cost = []
    for nl, bp, sp in zip(net_load, buy_price, sell_price):
        if nl >= 0:
            net_cost.append(nl * bp)
        else:
            net_cost.append(nl * sp)  # nl is negative, so this adds to profit
    
    return net_cost



bess_query = f'''
select * from bess where player_id = 1
'''

player_1_bess_data = cur.execute(bess_query).fetchall()

print(player_1_bess_data[0])

df = pd.read_sql_query(bess_query, conn)
print(df.head()) 


capacity = df['capacity'][0]
charge = df['charge'][0]
discharge = df['discharge'][0]
efficiency = df['efficiency'][0]
print(f"Capacity: {capacity}, Charge: {charge}, Discharge: {discharge}, Efficiency: {efficiency}")
exit()
                                


# quick plot for load, pv, net_load
import matplotlib.pyplot as plt
player_id = 1
load = fetch_timeseries(measurement="load", player_id=player_id)
pv = fetch_timeseries(measurement="pv", player_id=player_id)
net_load = [l - p for l, p in zip(load, pv)]
net_cost = generate_net_cost(player_id=player_id)

plt.figure(figsize=(12, 6))
plt.plot(load, label="Load", color="blue")
plt.plot(pv, label="PV", color="orange")
plt.plot(net_load, label="Net Load", color="green")
plt.plot(net_cost, label="Net Cost", color="red")
plt.xlabel("Time (15-min intervals)")
plt.ylabel("Power (kW)")
plt.title(f"Player {player_id} Load, PV, Net Load, and Net Cost")
plt.legend()
plt.grid()
plt.show()
