from dataclasses import dataclass, field

from src.simulation.scenarios.scenario import Scenario, default_scenario
from src.simulation.controllers.base_controller import BaseController


@dataclass
class RunContext:
    controller: BaseController
    scenario: Scenario = field(default_factory=lambda: default_scenario)
    run_id: str | None = None
    start_time: int = 1

    # ensure read-only after creation
    def __post_init__(self):
        # freeze after creation to prevent accidental mutation during a run
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("RunContext is read-only after creation")
        super().__setattr__(name, value)
