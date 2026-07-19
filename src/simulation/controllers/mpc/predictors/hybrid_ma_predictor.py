from __future__ import annotations

from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.simulation.controllers.mpc.predictors.hybrid_ma import (
    predict_base_load,
    predict_pv_gen,
    predict_ev_status,
    predict_ev_load,
    predict_ev_max_charge,
    predict_ev_buy_price,
    predict_buy_price,
    predict_sell_price,
)
from src.simulation.household import Household


class HybridMAPredictor(BasePredictor):
    """Hybrid MA predictor assembled from profile-specific helper functions.

    Tunables are intentionally small:
    - ``window_size``: moving-average window used for base load and PV
    - ``persistence_range``: number of initial steps that keep the last observed value
    - ``conf_interval_frct``: width factor for point-forecast bands
    """

    def __init__(
        self,
        window_size: int = 96,
        persistence_range: int = 1,
        conf_interval_frct: float = 0.1,
    ):
        self.window_size = max(1, int(window_size))
        self.persistence_range = max(0, int(persistence_range))
        self.interval_width_fraction = max(0.0, float(conf_interval_frct))

    def predict(self, household: Household, horizon: int) -> dict[str, list[float]]:
        horizon = max(0, int(horizon))

        base_load = predict_base_load(
            household,
            horizon,
            interval_width_fraction=self.interval_width_fraction,
        )
        pv_gen = predict_pv_gen(
            household,
            horizon,
            interval_width_fraction=self.interval_width_fraction,
        )
        ev_status = predict_ev_status(household, horizon)
        ev_load = predict_ev_load(household, horizon, ev_status)
        buy_price = predict_buy_price(household, horizon)
        sell_price = predict_sell_price(household, horizon)
        ev_buy_price = predict_ev_buy_price(household, horizon, ev_status)

        ev_max_charge = predict_ev_max_charge(household, horizon, ev_status)

        prediction: dict[str, list[float]] = {}
        prediction.update(base_load)
        prediction.update(pv_gen)
        prediction.update(ev_load)
        prediction.update(ev_status)
        prediction.update(buy_price)
        prediction.update(sell_price)
        prediction.update(ev_buy_price)
        prediction.update(ev_max_charge)

        return prediction
