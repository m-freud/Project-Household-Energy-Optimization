from __future__ import annotations
from abc import ABC, abstractmethod
from src.simulation.household import Household
from src.simulation.scenarios.scenario import Scenario


class BasePredictor(ABC):
    """Interface for predictors used by the MPC controller."""

    @abstractmethod
    def predict(
        self,
        household: Household,
        horizon: int,
    ) -> dict:
        """Return future household profiles for the next planning horizon.

        The returned payload should expose the same structure for every predictor,
        including any exogenous profiles that the MPC controller may need. At a
        minimum this should cover the household inputs that are updated each step,
        such as base load, PV generation, EV loads, EV availability, prices, and
        any other future signals that are relevant to the optimization problem.
        """
        raise NotImplementedError

    def predict_ev_status(self, household: Household, horizon: int) -> dict:
        raise NotImplementedError

    def predict_base_load(self, household: Household, horizon: int, ev_status_pred: dict) -> dict:
        raise NotImplementedError

    def predict_pv_gen(self, household: Household, horizon: int) -> dict:
        raise NotImplementedError
