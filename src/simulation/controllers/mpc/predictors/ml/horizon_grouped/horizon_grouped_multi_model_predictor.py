from __future__ import annotations

from typing import Generic

from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.simulation.controllers.mpc.predictors.ml.horizon_grouped.helpers.base_load import (
    predict_base_load,
)
from src.simulation.controllers.mpc.predictors.ml.horizon_grouped.helpers.ev_status import (
    predict_ev_status,
)
from src.simulation.controllers.mpc.predictors.ml.horizon_grouped.helpers.pv_gen import (
    predict_pv_gen,
)
from src.simulation.controllers.mpc.predictors.ml.model_interface import (
    TClassifier,
    TRegressor,
)
from src.simulation.controllers.mpc.predictors.shared import (
    predict_buy_price_home,
    predict_ev_buy_price,
    predict_ev_load,
    predict_ev_max_charge,
    predict_sell_price_home,
)
from src.simulation.household import Household


class HorizonGroupedMultiModelPredictor(
    BasePredictor,
    Generic[TRegressor, TClassifier],
):
    def __init__(
        self,
        base_load_model_bank: dict[str, TRegressor],
        pv_gen_model_bank: dict[str, TRegressor],
        ev1_model_bank: dict[str, TClassifier],
        ev2_model_bank: dict[str, TClassifier],
    ) -> None:
        self.base_load_model_bank = base_load_model_bank
        self.pv_gen_model_bank = pv_gen_model_bank
        self.ev1_model_bank = ev1_model_bank
        self.ev2_model_bank = ev2_model_bank

    def predict_ev_status(
        self,
        household: Household,
        horizon: int,
        ev_key: str | None = None,
    ) -> dict[str, list[int]]:
        return predict_ev_status(
            model_bank_ev1=self.ev1_model_bank,
            model_bank_ev2=self.ev2_model_bank,
            household=household,
            horizon=horizon,
            ev_key=ev_key,
        )

    def predict_base_load(
        self,
        household: Household,
        horizon: int,
        ev_status_pred: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        _ = ev_status_pred
        return predict_base_load(
            model_bank=self.base_load_model_bank,
            household=household,
            horizon=horizon,
        )

    def predict_pv_gen(
        self,
        household: Household,
        horizon: int,
    ) -> dict[str, list[float]]:
        return predict_pv_gen(
            model_bank=self.pv_gen_model_bank,
            household=household,
            horizon=horizon,
        )

    def predict(self, household: Household, horizon: int) -> dict[str, list]:
        ev_status = self.predict_ev_status(household, horizon)
        base_load = self.predict_base_load(
            household,
            horizon,
            ev_status_pred=ev_status,
        )
        pv_gen = self.predict_pv_gen(household, horizon)
        buy_price = predict_buy_price_home(household, horizon)
        sell_price = predict_sell_price_home(household, horizon)
        grid_prices = {
            "buy_price": buy_price["buy_price"],
            "sell_price": sell_price["sell_price"],
        }

        prediction: dict[str, list] = {}
        prediction.update(base_load)
        prediction.update(pv_gen)
        prediction.update(ev_status)
        prediction.update(predict_ev_load(household, horizon, ev_status))
        prediction.update(
            predict_ev_buy_price(
                household,
                horizon,
                ev_status,
                grid_prices=grid_prices,
            )
        )
        prediction.update(predict_ev_max_charge(household, horizon, ev_status))
        prediction.update(buy_price)
        prediction.update(sell_price)
        return prediction
