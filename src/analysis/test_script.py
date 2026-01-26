# paste this to enable src. imports

from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.simulation.main import init_simulation

simulation = init_simulation()
# household_1 = simulation.create_household(1)
# print(household_1.has_bess)

from simulation.policies.targeted_greedy import targeted_greedy

h1 = simulation.run_household(1, policy=targeted_greedy, start_time=0)
print(h1.has_bess)