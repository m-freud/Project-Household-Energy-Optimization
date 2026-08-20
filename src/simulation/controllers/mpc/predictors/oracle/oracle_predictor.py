from __future__ import annotations
from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.simulation.household import Household


class OraclePredictor(BasePredictor):
    """Predictor that uses the household's existing profile data as future forecasts."""

    def _pad_to_horizon(self, series: list, horizon: int, default: float = 0.0) -> list:
        if len(series) < horizon:
            fill_value = series[-1] if series else default
            series.extend([float(fill_value)] * (horizon - len(series)))
        return series

    def predict_ev_status(self, household: Household, horizon: int, ev_key: str|None = None) -> dict[str, list[int]]:
        profiles = household.oracle_profiles
        start = max(0, min(len(profiles.get("ev1_at_home", [])), household.current_timestep - 1))

        predictions = {
            "ev1_at_home": profiles.get("ev1_at_home", [])[start:start + horizon], # len 96
            "ev1_at_charging_station": profiles.get("ev1_at_charging_station", [])[start:start + horizon],
            "ev2_at_home": profiles.get("ev2_at_home", [])[start:start + horizon],
            "ev2_at_charging_station": profiles.get("ev2_at_charging_station", [])[start:start + horizon],
        }

        if ev_key in ["ev1", "ev2"]:
            predictions = {
                f"{ev_key}_at_home": predictions[f"{ev_key}_at_home"],
                f"{ev_key}_at_charging_station": predictions[f"{ev_key}_at_charging_station"],
            }

        for key, values in list(predictions.items()):
            predictions[key] = self._pad_to_horizon(values, horizon, default=0)

        return predictions

    def predict_base_load(self, household: Household, horizon: int, ev_status_pred: dict|None = None) -> dict[str, list[float]]:
        _ = ev_status_pred  # just for compatibility with ModularPredictor
        profiles = household.oracle_profiles
        start_index = max(0, min(len(profiles.get("base_load", [])), household.current_timestep - 1))
        pred = {"base_load": profiles.get("base_load", [])[start_index:start_index + horizon]}

        pred["base_load"] = self._pad_to_horizon(pred["base_load"], horizon, default=0.0)
        return pred


    def predict_pv_gen(self, household: Household, horizon: int) -> dict[str, list[float]]:
        profiles = household.oracle_profiles
        start_index = max(0, min(len(profiles.get("pv_gen", [])), household.current_timestep - 1))
        pred = {"pv_gen": profiles.get("pv_gen", [])[start_index:start_index + horizon]}
        pred["pv_gen"] = self._pad_to_horizon(pred["pv_gen"], horizon, default=0.0)
        return pred

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
