
from src.simulation.simulation import Simulation
from src.simulation.policies.blind import no_control, random_control
from src.simulation.policies.greedy import basic_bess, basic_ev, basic_ev_bess, advanced_ev_bess
from src.simulation.policies.lookahead import lookahead_greedy, lookahead_greedy_adaptive
from src import connections
from src.simulation.requirements.charge_requirements import basic_charge_requirements
from src.simulation.policies.optimization import mpc_control, mpc_short_horizon, mpc_long_horizon, mpc_scipy, robust_mpc_control


if __name__ == "__main__":
    sqlite_conn = connections.create_sqlite_connection()
    influx_client = connections.create_influx_client()

    charge_requirements = basic_charge_requirements

    simulation = Simulation(
        sqlite_conn=sqlite_conn,
        influx_client=influx_client,
        charge_requirements=charge_requirements
    )

    # Run baseline policies
    # for policy in [no_control, basic_bess, basic_ev_bess]:
    #     simulation.run_all_households(policy=policy, start_time=0)

    # Run advanced greedy
    # simulation.run_all_households(policy=advanced_ev_bess, start_time=0)
    
    # Run lookahead policies
    # simulation.run_all_households(policy=lookahead_greedy, start_time=0)
    # simulation.run_all_households(policy=lookahead_greedy_adaptive, start_time=0)

    # Run optimization-based policies
    for policy in [robust_mpc_control]:
        simulation.run_all_households(policy=policy, start_time=0)

    if sqlite_conn:
        sqlite_conn.close()
    if influx_client:
        influx_client.close()
