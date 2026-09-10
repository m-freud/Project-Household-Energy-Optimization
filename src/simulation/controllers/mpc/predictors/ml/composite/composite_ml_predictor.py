
from __future__ import annotations

from typing import Generic

from src.simulation.household import Household
from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor

from src.simulation.controllers.mpc.predictors.shared import (
    predict_ev_load,
    predict_ev_max_charge,
    predict_ev_buy_price,
    predict_buy_price_home,
    predict_sell_price_home,
)

from src.simulation.controllers.mpc.predictors.ml.model_interface import (
    TRegressor,
    TClassifier,
)


class CompositeMLPredictor(BasePredictor):
    def __init__(
            self,
            model_bank,
            interpolation: str = "linear"
            ) -> None:
        self.model_bank = model_bank
        self.interpolation = interpolation

    def predict_ev_status(self, household: Household, horizon: int, ev_key: str|None = None) -> dict:
        return {}

    def predict_base_load(self, household: Household, horizon: int) -> dict:
        return {}

    def predict_pv_gen(self, household: Household, horizon: int) -> dict:
        return {}

    def predict(
        self,
        household: Household,
        horizon: int,
    ) -> dict:
        return {}