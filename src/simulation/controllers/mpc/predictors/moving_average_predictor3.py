from __future__ import annotations

from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.simulation.controllers.mpc.predictors.ma3 import (
    predict_base_load,
    predict_pv_gen,
    predict_ev_load,
    predict_ev_max_charge,
    predict_ev_buy_price,
    predict_ev_status,
    predict_buy_price,
    predict_sell_price,
)
from src.simulation.household import Household
from src.simulation.scenarios.scenario import Scenario


class MovingAveragePredictor3(BasePredictor):
    """Modular MA3 predictor scaffold.

    This class composes lower-level sub-predictors from predictors/ma3:
    - house_profiles: base_load, pv_gen (+ optional interval bands)
    - ev_profiles: ev_load, ev_status, ev_max_charge
    - price_profiles: buy_price/sell_price and EV buy-price composition
    """

    def __init__(
        self,
        short_window_size: int = 7,
        long_window_size: int = 48,
        short_weight: float = 0.7,
        interval_width_fraction: float = 0.1,
        persistence_mode: str = "exponential",
        persistence_horizon: int = 8,
        persistence_constant_alpha: float = 0.5,
    ):
        self.short_window_size = max(1, int(short_window_size))
        self.long_window_size = max(self.short_window_size, int(long_window_size))
        self.short_weight = min(1.0, max(0.0, float(short_weight)))
        self.interval_width_fraction = max(0.0, float(interval_width_fraction))
        self.persistence_mode = str(persistence_mode)
        self.persistence_horizon = max(1, int(persistence_horizon))
        self.persistence_constant_alpha = min(1.0, max(0.0, float(persistence_constant_alpha)))

    def predict(self, household: Household, scenario: Scenario, horizon: int) -> dict:
        _ = (scenario,)
        horizon = max(0, int(horizon))

        base_load = predict_base_load(
            household,
            horizon,
            short_window=self.short_window_size,
            long_window=self.long_window_size,
            short_weight=self.short_weight,
            interval_width_fraction=self.interval_width_fraction,
            persistence_mode=self.persistence_mode,
            persistence_horizon=self.persistence_horizon,
            persistence_constant_alpha=self.persistence_constant_alpha,
        )
        pv_gen = predict_pv_gen(
            household,
            horizon,
            short_window=self.short_window_size,
            long_window=self.long_window_size,
            short_weight=self.short_weight,
            interval_width_fraction=self.interval_width_fraction,
            persistence_mode=self.persistence_mode,
            persistence_horizon=self.persistence_horizon,
            persistence_constant_alpha=self.persistence_constant_alpha,
        )
        ev_status = predict_ev_status(household, horizon)
        ev_load = predict_ev_load(
            household,
            horizon,
            ev_status,
            short_window=self.short_window_size,
            long_window=self.long_window_size,
            short_weight=self.short_weight,
        )
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
