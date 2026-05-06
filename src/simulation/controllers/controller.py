# paste this to enable src. imports

from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.simulation.household import Household


class Controller: # parent cclass TBD
    def __init__(self, name: str):
        self.name = name

    def step(self, household: Household):
        raise NotImplementedError("Controller step method must be implemented by subclasses")
    
