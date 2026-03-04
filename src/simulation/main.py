# paste this to enable src. imports

from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))


from src.simulation.simulation import Simulation
from src.simulation.policies.naive_linear import make_naive_linear_policy
from src.sqlite_connection import sqlite_conn
from src.simulation.scenarios.scenario import default_scenario


if __name__ == "__main__":
    # Create a simulation instance
    sim = Simulation(sqlite_conn)

    # Load the scenario
    scenario = default_scenario

    policies = [
        make_naive_linear_policy(urgency=1.0, delay=0.0),
        make_naive_linear_policy(urgency=0.0, delay=0.0),
        make_naive_linear_policy(urgency=0.0, delay=1.0),
        make_naive_linear_policy(urgency=0.5, delay=0.5),
    ]

    for policy in policies:
        sim.run_all_households(policy=policy, scenario=scenario, start_time=0)
