from __future__ import annotations

from functools import lru_cache

from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.simulation.controllers.mpc.predictors.hybrid_ma import (
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
from src.sqlite_connection import load_source_avg_profile


@lru_cache(maxsize=8)
def _load_source_avg_curve(table_name: str) -> list[float]:
    df = load_source_avg_profile(table_name)
    if df.empty:
        return []
    return [float(value) for value in df["value"].tolist()]


class HybridMAController(BasePredictor):
    """Modular Hybrid MA predictor scaffold.

    This class composes lower-level sub-predictors from predictors/hybrid_ma_controller:
    - house_profiles: base_load, pv_gen (+ optional interval bands)
    - ev_profiles: ev_load, ev_status, ev_max_charge
    - price_profiles: buy_price/sell_price and EV buy-price composition
    """

    def __init__(
        self,
        short_window_size: int = 7,
        long_window_size: int = 48,
        short_weight: float = 0.7,
        conf_interval_frct: float = 0.1,
        persistence_mode: str = "exponential",
        persistence_range: int = 8,
        persistence_constant_alpha: float = 0.5,
        trend_weight: float = 0.0,
        trend_window: int = 4,
        trend_range: int = 4,
        source_average_beta: float = 0.0,
        source_average_beta_base_load: float | None = None,
        source_average_beta_pv_gen: float | None = None,
    ):
        self.short_window_size = max(1, int(short_window_size))
        self.long_window_size = max(self.short_window_size, int(long_window_size))
        self.short_weight = min(1.0, max(0.0, float(short_weight)))
        self.interval_width_fraction = max(0.0, float(conf_interval_frct))
        self.persistence_mode = str(persistence_mode)
        self.persistence_range = max(1, int(persistence_range))
        self.persistence_constant_alpha = min(1.0, max(0.0, float(persistence_constant_alpha)))
        self.trend_weight = min(1.0, max(0.0, float(trend_weight)))
        self.trend_window = max(2, int(trend_window))
        self.trend_range = max(1, int(trend_range))
        self.source_average_beta = min(1.0, max(0.0, float(source_average_beta)))
        self.source_average_beta_base_load = (
            self.source_average_beta
            if source_average_beta_base_load is None
            else min(1.0, max(0.0, float(source_average_beta_base_load)))
        )
        self.source_average_beta_pv_gen = (
            self.source_average_beta
            if source_average_beta_pv_gen is None
            else min(1.0, max(0.0, float(source_average_beta_pv_gen)))
        )

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
            persistence_range=self.persistence_range,
            persistence_constant_alpha=self.persistence_constant_alpha,
            trend_weight=self.trend_weight,
            trend_window=self.trend_window,
            trend_range=self.trend_range,
        )
        pv_gen = predict_pv_gen(
            household,
            horizon,
            short_window=self.short_window_size,
            long_window=self.long_window_size,
            short_weight=self.short_weight,
            interval_width_fraction=self.interval_width_fraction,
            persistence_mode=self.persistence_mode,
            persistence_range=self.persistence_range,
            persistence_constant_alpha=self.persistence_constant_alpha,
            trend_weight=self.trend_weight,
            trend_window=self.trend_window,
            trend_range=self.trend_range,
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

        profile_betas = {
            "base_load": self.source_average_beta_base_load,
            "pv_gen": self.source_average_beta_pv_gen,
        }

        if any(beta > 0.0 for beta in profile_betas.values()):
            timestep = max(1, int(getattr(household, "current_timestep", 1)))
            start_idx = timestep - 1
            for profile_name in ("base_load", "pv_gen"):
                profile_beta = profile_betas[profile_name]
                if profile_beta <= 0.0:
                    continue
                source_avg = _load_source_avg_curve(profile_name)
                pred_series = [float(value) for value in prediction.get(profile_name, [])]
                if not source_avg or not pred_series:
                    continue

                blended: list[float] = []
                for idx, pred_value in enumerate(pred_series):
                    source_idx = start_idx + idx
                    if source_idx < len(source_avg):
                        source_value = float(source_avg[source_idx])
                        blended.append((1.0 - profile_beta) * pred_value + profile_beta * source_value)
                    else:
                        blended.append(pred_value)
                prediction[profile_name] = blended

        return prediction
