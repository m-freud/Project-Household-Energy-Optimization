
from src.simulation.simulation import Simulation
from src.simulation.policies.blind import no_control
from src.simulation.policies.greedy import advanced_ev_bess
from src import connections
from src.simulation.requirements.charge_requirements import basic_charge_requirements


if __name__ == "__main__":
    sqlite_conn = connections.create_sqlite_connection()
    influx_client = connections.create_influx_client()

    charge_requirements = basic_charge_requirements

    simulation = Simulation(
        sqlite_conn=sqlite_conn,
        influx_client=influx_client,
        charge_requirements=charge_requirements
    )

    simulation.run_all_households(policy=advanced_ev_bess, start_time=0)

    if sqlite_conn:
        sqlite_conn.close()
    if influx_client:
        influx_client.close()
