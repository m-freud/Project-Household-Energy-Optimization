from src.simulation.scenarios.scenario import Scenario
from src.simulation.controllers.base_controller import BaseController
from src.simulation.household import Household


class StepwiseController(BaseController):
    # BaseController but with step function
    # does not predict future, lives in the present, only uses current states
    def __init__(self, name: str, step_function: callable):
        super().__init__(name)
        self.step_function = step_function

    def set_controls(self, household: Household, scenario: Scenario, *args, **kwargs):
        return self.step_function(household, scenario, *args, **kwargs)
