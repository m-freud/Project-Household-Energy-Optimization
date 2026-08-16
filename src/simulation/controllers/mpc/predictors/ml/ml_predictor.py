
from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from src.simulation.household import Household
from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.simulation.controllers.mpc.predictors.ml.helpers import (
    predict_ev_status,
    predict_base_load,
    predict_pv_gen,
)
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
    PredictorModelBankLike,
)


class MLPredictor(BasePredictor, Generic[TRegressor, TClassifier]):
    """Base class for ML predictors (XGB, RF, Ridge, or more).
    Accepts PredictorModelBankLike that can be constructed from any ML model with a predict() function
    """

    def __init__(self, model_bank: PredictorModelBankLike[TRegressor, TClassifier]):
        self.model_bank = model_bank


    def predict(self, household: Household, horizon: int) -> dict:
        # select models based on household and metric
        household_id = household.player_id
        base_load_model: TRegressor = self.model_bank.base_load_model_bank.get_predictor_model(household_id)
        ev1_status_model: TClassifier = self.model_bank.ev1_status_model_bank.get_predictor_model(household_id)
        ev2_status_model: TClassifier = self.model_bank.ev2_status_model_bank.get_predictor_model(household_id)

        # use models to predict each metric
        prediction: dict[str, list] = {}

        ev_status = predict_ev_status(
            model_ev1=ev1_status_model,
            model_ev2=ev2_status_model,
            household=household,
            horizon=horizon,
        )
        base_load = predict_base_load(
            model=base_load_model,
            household=household,
            horizon=horizon,
            predicted_ev_status=ev_status,
        )
        if household.has_pv:
            pv_gen_model: TRegressor = self.model_bank.pv_gen_model_bank.get_predictor_model(household_id)
            pv_gen = predict_pv_gen(
                model=pv_gen_model,
                household=household,
                horizon=horizon,
            )
        else:
            pv_gen = {
                "pv_gen": [0.0] * horizon,
                "pv_gen_lb": [0.0] * horizon,
                "pv_gen_ub": [0.0] * horizon,
            }
        
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