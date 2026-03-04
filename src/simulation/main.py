# paste this to enable src. imports

from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))


from src.simulation.simulation import Simulation
from src.simulation.step_functions.basic_examples import no_control
from src.simulation.step_functions.naive_linear_satisfaction import naive_linear_satisfaction
from src.sqlite_connection import sqlite_conn
from src.simulation.scenarios.scenario import default_scenario


if __name__ == "__main__":
    # Create a simulation instance
    sim = Simulation(sqlite_conn)

    # Load the scenario
    scenario = default_scenario

    # Run the simulation for all households in the scenario
    # sim.run_all_households(policy=no_control, scenario=scenario, start_time=0)
    sim.run_all_households(policy=naive_linear_satisfaction, scenario=scenario, start_time=0)
