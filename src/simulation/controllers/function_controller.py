# paste this to enable src. imports

from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.controllers.base_controller import BaseController
from src.simulation.household import Household


class FunctionController(BaseController):
    def __init__(self, name: str, step_function: callable):
        super().__init__(name)
        self.step_function = step_function

    def set_controls(self, household: Household, *args, **kwargs):
        return self.step_function(household, *args, **kwargs)
