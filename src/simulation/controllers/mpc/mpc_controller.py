import cvxpy as cp

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
    def __init__(self, name: str, horizon: int = 96):
        super().__init__(name)
        self.horizon = int(horizon)

    def set_controls(self, household: Household, scenario: Scenario, predictor: str='oracle', *args, **kwargs):
        _ = (household, scenario, args, kwargs)

        # 1. get current states for the household

        # 2. get predictions for the next 24 hours
        # make optimization problem and solve it

        if predictor == 'oracle':
            predictions = household.oracle_profiles.get("predictions", {})
        else:
            print('No other predictor available yet, TBD, use oracle instead')
            return

        #3. define optimization problem

        bess_power = cp.Variable(self.horizon, bounds=(-household.bess.max_discharge, household.bess.max_charge)) if household.bess else None
        ev1_power = cp.Variable(self.horizon, bounds=(0, household.ev1.max_charge)) if household.ev1 else None
        ev2_power = cp.Variable(self.horizon, bounds=(0, household.ev2.max_charge)) if household.ev2 else None

        constraints = []

        # 4. solve optimization problem

        # 5. return the first control action

        return {
            "bess_power": 0.0,
            "ev1_power": 0.0,
            "ev2_power": 0.0,
        }
