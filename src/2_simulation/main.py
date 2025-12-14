from Simulation import Simulation
from db_connection import sql_conn, influx_query_api

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

