import src.connections as connections
from src.simulation.Simulation import Simulation
from src.simulation.policies.rule_based import basic_ev_bess
from src.connections import create_influx_client, create_sqlite_connection


if __name__ == "__main__":
    sqlite_conn = connections.create_sqlite_connection()
    influx_client = connections.create_influx_client()

    simulation = Simulation(
        sqlite_conn=sqlite_conn,
        influx_client=influx_client
    )

    player_id = 1

    simulation.run_all_households(policy=basic_ev_bess, start_time=0)

    if sqlite_conn:
        sqlite_conn.close()
    if influx_client:
        influx_client.close()
