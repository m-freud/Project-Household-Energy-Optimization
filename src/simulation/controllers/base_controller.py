from abc import ABC, abstractmethod
from src.runtime_config import RuntimeConfig
from src.simulation.household import Household


class BaseController(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def set_controls(self, household: Household, *args, **kwargs):
        raise NotImplementedError("Controller set_controls method must be implemented by subclasses")
