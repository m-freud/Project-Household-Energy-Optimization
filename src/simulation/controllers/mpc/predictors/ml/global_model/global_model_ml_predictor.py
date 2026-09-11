from __future__ import annotations

from typing import Generic

from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.simulation.controllers.mpc.predictors.ml.global_model.helpers.base_load import predict_base_load
from src.simulation.controllers.mpc.predictors.ml.global_model.helpers.ev_status import predict_ev_status
from src.simulation.controllers.mpc.predictors.ml.global_model.helpers.pv_gen import predict_pv_gen
from src.simulation.controllers.mpc.predictors.ml.model_interface import TClassifier, TRegressor
from src.simulation.controllers.mpc.predictors.shared import (
    predict_buy_price_home,
    predict_ev_buy_price,
    predict_ev_load,
    predict_ev_max_charge,
    predict_sell_price_home,
)
from src.simulation.household import Household


class GlobalModelMLPredictor(BasePredictor, Generic[TRegressor, TClassifier]):
    def __init__(
        self,
        base_load_model: TRegressor,
        pv_gen_model: TRegressor,
        ev1_model: TClassifier,
        ev2_model: TClassifier,
        prediction_horizons: list[int] | tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
        base_load_interpolation: str = "linear",
        pv_gen_interpolation: str = "linear",
        ev1_interpolation: str = "recursive",
        ev2_interpolation: str = "recursive",
    ) -> None:
        self.base_load_model = base_load_model
        self.pv_gen_model = pv_gen_model
        self.ev1_model = ev1_model
        self.ev2_model = ev2_model
        self.prediction_horizons = list(prediction_horizons)
        self.base_load_interpolation = base_load_interpolation
        self.pv_gen_interpolation = pv_gen_interpolation
        self.ev1_interpolation = ev1_interpolation
        self.ev2_interpolation = ev2_interpolation

    def predict_ev_status(self, household: Household, horizon: int, ev_key: str | None = None) -> dict:
        return predict_ev_status(
            model_ev1=self.ev1_model,
            model_ev2=self.ev2_model,
            prediction_horizons=self.prediction_horizons,
            household=household,
            horizon=horizon,
            ev_key=ev_key,
            ev1_interpolation=self.ev1_interpolation,
            ev2_interpolation=self.ev2_interpolation,
        )

    def predict_base_load(self, household: Household, horizon: int) -> dict:
        return predict_base_load(
            model=self.base_load_model,
            prediction_horizons=self.prediction_horizons,
            household=household,
            horizon=horizon,
            interpolation=self.base_load_interpolation,
        )

    def predict_pv_gen(self, household: Household, horizon: int) -> dict:
        return predict_pv_gen(
            model=self.pv_gen_model,
            prediction_horizons=self.prediction_horizons,
            household=household,
            horizon=horizon,
            interpolation=self.pv_gen_interpolation,
        )

    def predict(self, household: Household, horizon: int) -> dict:
        ev_status = self.predict_ev_status(household, horizon)
        base_load = self.predict_base_load(household, horizon)
        pv_gen = self.predict_pv_gen(household, horizon)
        buy_price = predict_buy_price_home(household, horizon)
        sell_price = predict_sell_price_home(household, horizon)
        grid_prices = {"buy_price": buy_price["buy_price"], "sell_price": sell_price["sell_price"]}

        prediction: dict[str, list] = {}
        prediction.update(base_load)
        prediction.update(pv_gen)
        prediction.update(ev_status)
        prediction.update(predict_ev_load(household, horizon, ev_status))
        prediction.update(predict_ev_buy_price(household, horizon, ev_status, grid_prices=grid_prices))
        prediction.update(predict_ev_max_charge(household, horizon, ev_status))
        prediction.update(buy_price)
        prediction.update(sell_price)
        return prediction
