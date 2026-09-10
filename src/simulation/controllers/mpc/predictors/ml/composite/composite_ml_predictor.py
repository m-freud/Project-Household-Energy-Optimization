
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

from src.simulation.controllers.mpc.predictors.ml.composite.helpers.ev_status import predict_ev_status
from src.simulation.controllers.mpc.predictors.ml.composite.helpers.base_load import predict_base_load
from src.simulation.controllers.mpc.predictors.ml.composite.helpers.pv_gen import predict_pv_gen


class CompositeMLPredictor(BasePredictor, Generic[TRegressor, TClassifier]):
    def __init__(
            self,
            base_load_model_bank: dict[int, TRegressor],
            pv_gen_model_bank: dict[int, TRegressor],
            ev1_model_bank: dict[int, TClassifier],
            ev2_model_bank: dict[int, TClassifier],

            base_load_interpolation: str = "linear",
            pv_gen_interpolation: str = "linear",
            ev1_interpolation: str = "recursive",
            ev2_interpolation: str = "recursive",
            ) -> None:
        
        self.base_load_model_bank = base_load_model_bank
        self.pv_gen_model_bank = pv_gen_model_bank
        self.ev1_model_bank = ev1_model_bank
        self.ev2_model_bank = ev2_model_bank
        self.base_load_interpolation = base_load_interpolation
        self.pv_gen_interpolation = pv_gen_interpolation
        self.ev1_interpolation = ev1_interpolation
        self.ev2_interpolation = ev2_interpolation

    def predict_ev_status(self, household: Household, horizon: int, ev_key: str|None = None) -> dict:
        if self.ev1_model_bank is None and self.ev2_model_bank is None:
            return {
                "ev1_at_home": [0] * horizon,
                "ev1_at_charging_station": [0] * horizon,
                "ev2_at_home": [0] * horizon,
                "ev2_at_charging_station": [0] * horizon,
            }
        
        return predict_ev_status( 
            model_bank_ev1=self.ev1_model_bank,
            model_bank_ev2=self.ev2_model_bank,
            household=household,
            horizon=horizon,
            ev_key=ev_key,
        )

    def predict_base_load(self, household: Household, horizon: int) -> dict:
        if self.base_load_model_bank is None:
            return {
                "base_load": [0.0] * horizon,
                "base_load_lb": [0.0] * horizon,
                "base_load_ub": [0.0] * horizon,
            }

        return predict_base_load(
            model_bank=self.base_load_model_bank,
            household=household,
            horizon=horizon,
            interpolation=self.base_load_interpolation,
        )

    def predict_pv_gen(self, household: Household, horizon: int) -> dict:
        if not household.has_pv or self.pv_gen_model_bank is None:
            return {
                "pv_gen": [0.0] * horizon,
                "pv_gen_lb": [0.0] * horizon,
                "pv_gen_ub": [0.0] * horizon,
            }
        
        return predict_pv_gen(
            model_bank=self.pv_gen_model_bank,
            household=household,
            horizon=horizon,
            interpolation=self.pv_gen_interpolation,
        )


    def predict(self, household: Household, horizon: int) -> dict:
        # use models to predict each metric
        prediction: dict[str, list] = {}

        ev_status = self.predict_ev_status(household, horizon)
        base_load = self.predict_base_load(household, horizon)
        pv_gen = self.predict_pv_gen(household,horizon)
        
        ev_load = predict_ev_load(household, horizon, ev_status)
        buy_price = predict_buy_price_home(household, horizon)
        sell_price = predict_sell_price_home(household, horizon)
        grid_prices = {
            "buy_price": buy_price["buy_price"],
            "sell_price": sell_price["sell_price"],
        }
        ev_buy_price = predict_ev_buy_price(household, horizon, ev_status, grid_prices=grid_prices)
        ev_max_charge = predict_ev_max_charge(household, horizon, ev_status)


        prediction.update(base_load)
        prediction.update(pv_gen)
        prediction.update(ev_status) # int

        prediction.update(ev_load)
        prediction.update(ev_buy_price)
        prediction.update(ev_max_charge)

        prediction.update(buy_price)
        prediction.update(sell_price)
        
        return prediction
