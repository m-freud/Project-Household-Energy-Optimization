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
from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor


class MPCController(BaseController):
    def __init__(
        self,
        name: str,
        household: Household,
        scenario: Scenario,
        horizon: int = 96,
        predictor: BasePredictor | None = None,
    ):
        super().__init__(name)
        self.household = household
        self.scenario = scenario
        self.horizon = int(horizon)
        self.predictor = predictor

        self.bess_bounds = [-self.household.bess.max_discharge, self.household.bess.max_charge] if self.household.bess else None
        self.ev1_bounds = [0, self.household.ev1.max_charge] if self.household.ev1 else None
        self.ev2_bounds = [0, self.household.ev2.max_charge] if self.household.ev2 else None

        self.bess_soc_targets = self.scenario.bess.soc_targets if self.household.bess else None
        self.ev1_soc_targets = self.scenario.ev1.soc_targets if self.household.ev1 else None
        self.ev2_soc_targets = self.scenario.ev2.soc_targets if self.household.ev2 else None


    def set_controls(self, household: Household, scenario: Scenario, predictor: str='oracle', *args, **kwargs):
        _ = (household, scenario, args, kwargs)

        # 1. get current states for the household
        #  a. essentials
        current_timestep = household.current_timestep
        bess_soc = household.bess_soc
        ev1_soc = household.ev1_soc
        ev2_soc = household.ev2_soc
        base_load = household.base_load
        pv_generation = household.pv_gen
        net_load = household.net_load

        # b. potentially optional
        buy_price = household.buy_price
        sell_price = household.sell_price
        ev1_at_home = household.ev1_at_home
        ev2_at_home = household.ev2_at_home
        ev1_at_charging_station = household.ev1_at_charging_station
        ev2_at_charging_station = household.ev2_at_charging_station
        ev1_load = household.ev1_load
        ev2_load = household.ev2_load


        # 2. get predictions for the next 24 hours

        if self.predictor is not None:
            predictions = self.predictor.predict(household, scenario, self.horizon)
        else:
            print('No predictor configured for MPC controller; using empty predictions')
            predictions = {}

        #3. define optimization problem

        # TODO use scenario device constraints here
        bess_power = cp.Variable(self.horizon, bounds=self.bess_bounds) if household.bess else None
        ev1_power = cp.Variable(self.horizon, bounds=self.ev1_bounds) if household.ev1 else None
        ev2_power = cp.Variable(self.horizon, bounds=self.ev2_bounds) if household.ev2 else None

        constraints = []

        # 4. solve optimization problem

        # 5. return the first control action

        return {
            "bess_power": 0.0,
            "ev1_power": 0.0,
            "ev2_power": 0.0,
        }
