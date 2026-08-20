from __future__ import annotations

from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.simulation.controllers.mpc.predictors.history_avg import (
    predict_base_load,
    predict_pv_gen,
    predict_ev_status,
    predict_ev_load,
    predict_ev_max_charge,
    predict_ev_buy_price,
    predict_buy_price_home,
    predict_sell_price_home,
)
from src.simulation.household import Household


class HistoryAveragePredictor(BasePredictor):
    """Cumulative average predictor assembled from profile-specific helper functions.

    Tunables are intentionally small:
    - ``window_size``: moving-average window used for base load and PV
    - ``conf_interval_frct``: width factor for point-forecast bands
    """

    def __init__(
        self,
        conf_interval_frct: float = 0.1,
    ):
        self.interval_width_fraction = max(0.0, float(conf_interval_frct))

    def predict_ev_status(self, household: Household, horizon: int, ev_key: str|None = None) -> dict[str, list[int]]:
        return predict_ev_status(household, horizon, ev_key=ev_key)

    def predict_base_load(self, household: Household, horizon: int, ev_status_pred: dict[str, list[int]] | None = None) -> dict[str, list[float]]:
        _ = ev_status_pred  # just for compatibility with ModularPredictor
        return predict_base_load(
            household,
            horizon,
            interval_width_fraction=self.interval_width_fraction,
        )

    def predict_pv_gen(self, household: Household, horizon: int) -> dict[str, list[float]]:
        return predict_pv_gen(
            household,
            horizon,
            interval_width_fraction=self.interval_width_fraction,
        )

    def predict(self, household: Household, horizon: int) -> dict[str, list[float]]:
        ev_status = self.predict_ev_status(household, horizon)
        base_load = self.predict_base_load(household, horizon)

        if household.has_pv:
            pv_gen = self.predict_pv_gen(household, horizon)
        else:
            pv_gen = {
                "pv_gen": [0.0] * horizon,
                "pv_gen_lb": [0.0] * horizon,
                "pv_gen_ub": [0.0] * horizon,
            }

        ev_load = predict_ev_load(household, horizon, ev_status)
        ev_buy_price = predict_ev_buy_price(household, horizon, ev_status)
        ev_max_charge = predict_ev_max_charge(household, horizon, ev_status)
        
        buy_price = predict_buy_price_home(household, horizon)
        sell_price = predict_sell_price_home(household, horizon)

        prediction: dict[str, list] = {}
        prediction.update(base_load)
        prediction.update(pv_gen)
        prediction.update(ev_load)
        prediction.update(ev_status)
        prediction.update(buy_price)
        prediction.update(sell_price)
        prediction.update(ev_buy_price)
        prediction.update(ev_max_charge)

        return prediction
