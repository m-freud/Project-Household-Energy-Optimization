
from src.simulation.simulation import Simulation
from src.simulation.policies.blind import no_control
from simulation.policies.basic_advanced_slop import advanced_ev_bess
from src import connections
from src.simulation.requirements.charge_requirements import half_full_by_midnight


def init_simulation(charge_requirements=half_full_by_midnight):
    sqlite_conn = connections.create_sqlite_connection()
    influx_client = connections.create_influx_client()

    simulation = Simulation(
        sqlite_conn=sqlite_conn,
        influx_client=influx_client,
        charge_requirements=charge_requirements
    )

    return simulation
    

if __name__ == "__main__":
    simulation = init_simulation()

    simulation.run_all_households(policy=advanced_ev_bess, start_time=0)

    if simulation.sqlite_conn:
        simulation.sqlite_conn.close()
    if simulation.influx_client:
        simulation.influx_client.close()