from __future__ import annotations

from collections.abc import Mapping

from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.simulation.household import Household
from src.simulation.controllers.mpc.predictors.shared import (
    predict_ev_load,
    predict_ev_max_charge,
    predict_ev_buy_price,
    predict_buy_price_home,
    predict_sell_price_home,
)


class ModularPredictor(BasePredictor):
    """Compose multiple predictors by routing selected targets to specific models.

    The default predictor produces a complete prediction dictionary. Any entry in
    ``target_predictors`` overrides that key with the value produced by the mapped
    predictor. This keeps the class compatible with all existing predictor classes
    as long as they implement ``predict(household, horizon)``.
    """
    def __init__(
        self,
        *,
        default_predictor: BasePredictor,
        target_predictors: Mapping[str, BasePredictor] | None = None,
    ):
        self.default_predictor = default_predictor
        self.target_predictors = dict(target_predictors or {})

    def predict(self, household: Household, horizon: int) -> dict:
        if "ev_status" in self.target_predictors:
            ev_status = self.target_predictors["ev_status"].predict_ev_status(household, horizon)
        else:
            ev_status = self.default_predictor.predict_ev_status(household, horizon)
            if "ev1_status" in self.target_predictors:
                ev1_status = self.target_predictors["ev1_status"].predict_ev_status(household, horizon, ev_key="ev1")
                ev_status.update(ev1_status)
            if "ev2_status" in self.target_predictors:
                ev2_status = self.target_predictors["ev2_status"].predict_ev_status(household, horizon, ev_key="ev2")
                ev_status.update(ev2_status)

        # if "ev1_status" in self.target_predictors.keys() or "ev2_status" in self.target_predictors.keys():
        #     pre_ev_status = self.target_predictors["ev_status"].predict_ev_status(household, horizon)
        #     ev_status = {}
        #     for ev_status_key in ["ev1_status", "ev2_status"]:
        #         if ev_status_key in self.target_predictors.keys():
        #             ev_status[ev_status_key] = pre_ev_status[ev_status_key]
        #         else:
        #             ev_status[ev_status_key] = self.default_predictor.predict_ev_status(household, horizon)

        # if "ev_status" in self.target_predictors.keys():
        #     ev_status = self.target_predictors["ev_status"].predict_ev_status(household, horizon)
        # else:
        #     ev_status = self.default_predictor.predict_ev_status(household, horizon)

        if "base_load" in self.target_predictors.keys():
            base_load = self.target_predictors["base_load"].predict_base_load(household, horizon, ev_status_pred=ev_status)
        else:
            base_load = self.default_predictor.predict_base_load(household, horizon, ev_status_pred=ev_status)

        if "pv_gen" in self.target_predictors.keys():
            pv_gen = self.target_predictors["pv_gen"].predict_pv_gen(household, horizon)
        else:
            pv_gen = self.default_predictor.predict_pv_gen(household, horizon)

        ev_load = predict_ev_load(household, horizon, ev_status)
        buy_price = predict_buy_price_home(household, horizon)
        sell_price = predict_sell_price_home(household, horizon)
        grid_prices = {
            "buy_price": buy_price["buy_price"],
            "sell_price": sell_price["sell_price"],
        }
        ev_buy_price = predict_ev_buy_price(household, horizon, ev_status, grid_prices=grid_prices)
        ev_max_charge = predict_ev_max_charge(household, horizon, ev_status)

        prediction: dict[str, list] = {}
        prediction.update(base_load)
        prediction.update(pv_gen)
        prediction.update(ev_status) # int

        prediction.update(ev_load)
        prediction.update(ev_buy_price)
        prediction.update(ev_max_charge)

        prediction.update(buy_price)
        prediction.update(sell_price)
        
        return prediction