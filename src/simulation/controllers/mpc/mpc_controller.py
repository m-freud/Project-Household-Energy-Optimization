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
    def __init__(self, name: str):
        super().__init__(name)

    def set_controls(self, household: Household, scenario: Scenario, *args, **kwargs):
        pass
