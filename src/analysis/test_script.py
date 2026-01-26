# paste this to enable src. imports

from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.simulation.main import init_simulation
from simulation.policies.targeted_greedy import targeted_greedy
from simulation.scenarios.example_scenarios import default_scenario
from src.analysis.plotting.household_plotter import plot_household

simulation = init_simulation()
# household_1 = simulation.create_household(1)
# print(household_1.has_bess)


h1 = simulation.run_household(1, policy=targeted_greedy, scenario=default_scenario, start_time=0)
print(h1.has_bess)

fig = plot_household(
    household_id=1,
    policy="targeted_greedy",
    scenario=default_scenario,
    fields=["net_load", "pv_gen", "bess_soc", "ev1_soc", "ev2_soc"],
    colors={"net_load": "blue", "pv_gen": "orange", "bess_soc": "green", "ev1_soc": "red", "ev2_soc": "purple"},
    title="Household 1 Energy Profile under Targeted Greedy Policy"
)

fig.show()