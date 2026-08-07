from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor

from src.simulation.controllers.mpc.predictors.xgboost import (
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

    def __init__(
            self,
            base_load_regressor: XGBRegressor, # let simulation handle the model imports
            pv_gen_regressor: XGBRegressor,
            ev_status_classifier: XGBClassifier
    ):
        self.predictors = {
            "base_load": base_load_regressor,
            "pv_gen": pv_gen_regressor,
            "ev_status": ev_status_classifier
        }


    def predict(self, household: Household, horizon: int) -> dict:
        prediction: dict[str, list[float]] = {}

        ev_status = predict_ev_status(
            model=self.predictors["ev_status"],
            household=household,
            horizon=horizon,
        )
        base_load = predict_base_load(
            model=self.predictors["base_load"],
            household=household,
            horizon=horizon,
            predicted_ev_status=ev_status,
        )
        pv_gen = predict_pv_gen(
            model=self.predictors["pv_gen"],
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
        prediction.update(ev_status)

        prediction.update(ev_load)
        prediction.update(ev_buy_price)
        prediction.update(ev_max_charge)

        prediction.update(buy_price)
        prediction.update(sell_price)
        
        return prediction
