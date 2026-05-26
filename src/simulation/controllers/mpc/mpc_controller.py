# paste this to enable src. imports

from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.scenarios.scenario import Scenario
from src.simulation.controllers.base_controller import BaseController
from src.simulation.household import Household


class MPCController(BaseController):
    def __init__(self, name: str, horizon: int = 24):
        super().__init__(name)
        self.horizon = int(horizon)

    def set_controls(self, household: Household, scenario: Scenario, *args, **kwargs):
        _ = (household, scenario, args, kwargs)
        return {
            "bess_power": 0.0,
            "ev1_power": 0.0,
            "ev2_power": 0.0,
        }
