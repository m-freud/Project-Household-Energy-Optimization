from __future__ import annotations
from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.simulation.household import Household


class OraclePredictor(BasePredictor):
    """Predictor that uses the household's existing profile data as future forecasts."""

    def predict(self, household: Household, horizon: int) -> dict:
        profiles = household.oracle_profiles

        prediction = {
            "base_load": profiles.get("base_load", []),
            "pv_gen": profiles.get("pv_gen", []),
            "ev1_load": profiles.get("ev1_load", []),
            "ev2_load": profiles.get("ev2_load", []),
            "ev1_at_home": profiles.get("ev1_at_home", []),
            "ev1_at_charging_station": profiles.get("ev1_at_charging_station", []),
            "ev2_at_home": profiles.get("ev2_at_home", []),
            "ev2_at_charging_station": profiles.get("ev2_at_charging_station", []),
            "buy_price": profiles.get("buy_price", []),
            "sell_price": profiles.get("sell_price", []),
            "ev1_buy_price": profiles.get("ev1_buy_price", []),
            "ev2_buy_price": profiles.get("ev2_buy_price", []),
            "ev1_max_charge": profiles.get("ev1_max_charge", []),
            "ev2_max_charge": profiles.get("ev2_max_charge", []),
        }

        start_index = max(0, min(len(profiles.get("base_load", [])), household.current_timestep - 1))
        for key, values in list(prediction.items()):
            if isinstance(values, list):
                if not values:
                    prediction[key] = []
                    continue
                prediction[key] = values[start_index:start_index + horizon]

        return prediction
