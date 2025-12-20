
from src.simulation.simulation import Simulation
from src.simulation.policies.basic import no_control
from src import connections


if __name__ == "__main__":
    sqlite_conn = connections.create_sqlite_connection()
    influx_client = connections.create_influx_client()

    simulation = Simulation(
        sqlite_conn=sqlite_conn,
        influx_client=influx_client
    )

    simulation.run_all_households(policy=no_control, start_time=0)

    if sqlite_conn:
        sqlite_conn.close()
    if influx_client:
        influx_client.close()
