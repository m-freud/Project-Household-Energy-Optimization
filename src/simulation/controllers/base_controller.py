# paste this to enable src. imports

from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.simulation.household import Household


class BasicController:
    # minimal controlller, instantiated  with step function, stateless
    def __init__(self, name: str, step_function: callable):
        self.name = name
        self.step_function = step_function

    def set_controls(self, household: Household, *args, **kwargs):
        return self.step_function(household, *args, **kwargs)
