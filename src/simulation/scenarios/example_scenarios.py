# paste this to enable src. imports

from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.scenarios.scenario import Scenario, DeviceRequirement


no_requirements = Scenario(name="no_requirements")

default_scenario = Scenario(
    name="default_scenario",
    ev1=DeviceRequirement(start_soc=0.2, target_soc=0.5, deadline=96),
    ev2=DeviceRequirement(start_soc=0.2, target_soc=0.5, deadline=96),
    bess=DeviceRequirement(start_soc=0.0, target_soc=0.5, deadline=96)
)