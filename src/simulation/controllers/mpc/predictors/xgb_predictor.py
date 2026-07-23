from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor


from src.simulation.controllers.mpc.predictors.xgboost import (
    predict_base_load,
    predict_pv_gen,
    predict_ev_status,
)

from src.simulation.controllers.mpc.predictors.hybrid_running_avg import (
    predict_ev_load,
    predict_ev_max_charge,
    predict_ev_buy_price,
    predict_buy_price_home,
    predict_sell_price_home,
)
from src.simulation.household import Household

class XGBPredictor(BasePredictor):
    """
    Part XGBoost, part helper functions like in HyRuAvg

    base_load: xgboost
    pv_gen: xgboost
    ev_status: phase 1: dynamic worst case; phase 2: xgboost
    ev_load: ev_status
    ev_max_charge: ev_status
    ev_buy_price: ev_status
    buy_price: lookup table
    sell_price: lookup table
    """

    def __init__(
            self,
    ):
        pass

    def predict(self, household: Household, horizon: int) -> dict:
        prediction: dict[str, list[float]] = {}

        base_load = predict_base_load(
            household,
            horizon,
        )
        pv_gen = predict_pv_gen(
            household,
            horizon,
        )
        ev_status = predict_ev_status(
            household,
            horizon,
        )
        
        ev_load = predict_ev_load(household, horizon, ev_status)
        ev_buy_price = predict_ev_buy_price(household, horizon, ev_status)
        ev_max_charge = predict_ev_max_charge(household, horizon, ev_status)

        buy_price = predict_buy_price_home(household, horizon)
        sell_price = predict_sell_price_home(household, horizon)


        prediction.update(base_load)
        prediction.update(pv_gen)
        prediction.update(ev_status)

        prediction.update(ev_load)
        prediction.update(ev_buy_price)
        prediction.update(ev_max_charge)

        prediction.update(buy_price)
        prediction.update(sell_price)
        
        return prediction
