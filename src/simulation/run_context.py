from dataclasses import dataclass, field
from typing import Callable

from src.simulation.household import Household
from src.simulation.scenarios.scenario import Scenario, default_scenario
from src.simulation.controllers.base_controller import BaseController
from src.sqlite_connection import sqlite_conn as _default_conn


def _next_run_id() -> str:
    try:
        cursor = _default_conn.cursor()
        result = cursor.execute(
            '''
            SELECT COALESCE(MAX(CAST(run_id AS INTEGER)), 0) + 1
            FROM results
            WHERE run_id IS NOT NULL
              AND TRIM(run_id) <> ''
              AND run_id NOT GLOB '*[^0-9]*'
            '''
        ).fetchone()
        return str(result[0])
    except Exception:
        return "1"


@dataclass
class RunContext:
    controller_factory: Callable[[Household, Scenario], BaseController] | None = None
    scenario: Scenario = field(default_factory=lambda: default_scenario)
    run_id: str = field(default="")
    start_time: int = 1
    controller_name: str | None = None

    # ensure read-only after creation
    def __post_init__(self):
        if not self.run_id:
            object.__setattr__(self, "run_id", _next_run_id())
        # freeze after creation to prevent accidental mutation during a run
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("RunContext is read-only after creation")
        super().__setattr__(name, value)
