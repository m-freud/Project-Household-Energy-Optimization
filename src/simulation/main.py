import src.connections as connections
from src.simulation.Simulation import Simulation


if __name__ == "__main__":
    sqlite_conn = connections.create_sqlite_connection()
    influx_query_api = connections.get_influx_query_api()

    simulation = Simulation(
        sqlite_conn=sqlite_conn,
        influx_query_api=influx_query_api
    )
    # fetch everything.
    # you now have <250> Households with:
    # - Player (has load, ev1, ev2 profiles)
    # - PV (if any) (has pv profile)
    # - ESS (if any) (has ess characteristics)
    # 

    simulation.run_household()
    # this will simulate each strategy for each household
    # and push the results to influxdb. done

    # you can now marvel at you results in grafana
    # or analyze them in a notebook!

    # make your dreams come true!
    # nothing is impossible!

