from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.simulation.controllers.mpc.predictors.xgboost_legacy import (
    predict_base_load,
    predict_pv_gen,
    predict_ev_status,
)

from src.simulation.controllers.mpc.predictors.shared import (
    predict_ev_load,
    predict_ev_max_charge,
    predict_ev_buy_price,
    predict_buy_price_home,
    predict_sell_price_home,
)
from src.simulation.household import Household
from xgboost import XGBClassifier, XGBRegressor
from typing import Generic, TypeVar

from dataclasses import dataclass

TModel = TypeVar("TModel", XGBRegressor, XGBClassifier)

@dataclass
class FoldModelBank(Generic[TModel]):
    models_by_fold: dict[str, TModel]
    id_to_fold: dict[int, str]

    def get_predictor_model(self, player_id: int) -> TModel:
        fold_id = self.id_to_fold[player_id]
        return self.models_by_fold[fold_id]

@dataclass
class PredictorModelBank:
    base_load_model_bank: FoldModelBank[XGBRegressor]
    pv_gen_model_bank: FoldModelBank[XGBRegressor]
    ev1_status_model_bank: FoldModelBank[XGBClassifier]
    ev2_status_model_bank: FoldModelBank[XGBClassifier]


class XGBPredictor(BasePredictor):
    """
    Part XGBoost, part helper functions like in RuAvg

    base_load: xgboost
    pv_gen: xgboost
    ev_status: xgboost
    ev_load: ev_status
    ev_max_charge: ev_status
    ev_buy_price: ev_status
    buy_price: lookup table
    sell_price: lookup table
    """

    def __init__(self,predictor_model_bank: PredictorModelBank):
        self.predictor_model_bank = predictor_model_bank


    def predict(self, household: Household, horizon: int) -> dict:
        # select models based on household and metric
        household_id = household.player_id
        base_load_model: XGBRegressor = self.predictor_model_bank.base_load_model_bank.get_predictor_model(household_id)
        pv_gen_model: XGBRegressor = self.predictor_model_bank.pv_gen_model_bank.get_predictor_model(household_id)
        ev1_status_model: XGBClassifier = self.predictor_model_bank.ev1_status_model_bank.get_predictor_model(household_id)
        ev2_status_model: XGBClassifier = self.predictor_model_bank.ev2_status_model_bank.get_predictor_model(household_id)


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
        pv_gen = predict_pv_gen(
            model=pv_gen_model,
            household=household,
            horizon=horizon,
        )
        
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
