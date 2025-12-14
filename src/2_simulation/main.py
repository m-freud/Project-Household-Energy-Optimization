import sys
import sqlite3
from pathlib import Path
from influxdb_client.client.influxdb_client import InfluxDBClient

# ROOT_DIR = Path(__file__).parent.parent.parent
# sys.path.append(str(ROOT_DIR))


from config import Config

print(Config.INFLUX_URL)

exit()
from Simulation import Simulation


import connections



if __name__ == "__main__":
    simulation = Simulation(sql_conn, influx_query_api, strategies)
    # fetch everything. (*EEEEVERYTHING!!!*)
    # you now have <250> Households with:
    # - Player (has load, ev1, ev2 profiles)
    # - PV (if any) (has pv profile)
    # - ESS (if any) (has ess characteristics)
    # 

    simulation.run()
    # this will simulate each strategy for each household
    # and the results to influxdb. done

