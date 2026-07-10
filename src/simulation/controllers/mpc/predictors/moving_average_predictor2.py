from __future__ import annotations

from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.simulation.household import Household
from src.simulation.scenarios.scenario import Scenario


class MovingAveragePredictor2(BasePredictor):
	"""Predict future profiles using a blended short and long moving average.

	The short window reacts quickly to recent changes while the long window
	establishes a slower baseline. The forecast remains causal and recursive: each
	new prediction is fed back into the next step.
	"""

	def __init__(
		self,
		short_window_size: int = 7,
		long_window_size: int = 48,
		short_weight: float = 0.7,
		binary_threshold: float = 0.5,
	):
		self.short_window_size = max(1, int(short_window_size))
		self.long_window_size = max(self.short_window_size, int(long_window_size))
		self.short_weight = min(1.0, max(0.0, float(short_weight)))
		self.long_weight = 1.0 - self.short_weight
		self.binary_threshold = float(binary_threshold)

	def predict(self, household: Household, scenario: Scenario, horizon: int) -> dict:
		_ = (scenario,)

		horizon = max(0, int(horizon))

		prediction = {
			"base_load": self._forecast_series(household, "base_load", horizon, default=household.base_load),
			"pv_gen": self._forecast_series(household, "pv_gen", horizon, default=household.pv_gen),
			"ev1_load": self._forecast_series(household, "ev1_load", horizon, default=household.ev1_load),
			"ev2_load": self._forecast_series(household, "ev2_load", horizon, default=household.ev2_load),
			"ev1_at_home": self._forecast_binary_series(household, "ev1_at_home", horizon, default=1.0 if household.ev1_at_home else 0.0),
			"ev1_at_charging_station": self._forecast_binary_series(household, "ev1_at_charging_station", horizon, default=1.0 if household.ev1_at_charging_station else 0.0),
			"ev2_at_home": self._forecast_binary_series(household, "ev2_at_home", horizon, default=1.0 if household.ev2_at_home else 0.0),
			"ev2_at_charging_station": self._forecast_binary_series(household, "ev2_at_charging_station", horizon, default=1.0 if household.ev2_at_charging_station else 0.0),
			"buy_price": self._forecast_series(household, "buy_price", horizon, default=household.buy_price),
			"sell_price": self._forecast_series(household, "sell_price", horizon, default=household.sell_price),
			"ev1_buy_price": self._forecast_series(household, "ev1_buy_price", horizon, default=getattr(household.ev1, "buy_price", household.buy_price)),
			"ev2_buy_price": self._forecast_series(household, "ev2_buy_price", horizon, default=getattr(household.ev2, "buy_price", household.buy_price)),
			"ev1_max_charge": self._constant_series(getattr(household.ev1, "max_charge", 0.0), horizon),
			"ev2_max_charge": self._constant_series(getattr(household.ev2, "max_charge", 0.0), horizon),
		}

		return prediction

	def _constant_series(self, value: float, horizon: int) -> list[float]:
		return [float(value)] * horizon

	def _history_values(self, household: Household, key: str) -> list[float]:
		history = household.history.get(key, {})
		if not history:
			return []
		return [float(history[timestep]) for timestep in sorted(history)]

	def _fallback_series(self, household: Household, key: str, horizon: int, default: float) -> list[float]:
		_ = (household, key)
		return [float(default)] * horizon

	def _seed_series(self, household: Household, key: str, default: float) -> list[float]:
		history = self._history_values(household, key)
		seed = history[-self.long_window_size :]

		if len(seed) >= self.long_window_size:
			return seed

		needed = self.long_window_size - len(seed)
		if seed:
			return [float(seed[0])] * needed + seed

		fallback = self._fallback_series(household, key, self.long_window_size, default)
		return fallback[:needed]

	def _forecast_series(self, household: Household, key: str, horizon: int, default: float) -> list[float]:
		if horizon <= 0:
			return []

		series = self._seed_series(household, key, default)
		forecast: list[float] = []

		for _ in range(horizon):
			short_window_values = series[-self.short_window_size :]
			long_window_values = series[-self.long_window_size :]

			short_average = sum(short_window_values) / len(short_window_values) if short_window_values else float(default)
			long_average = sum(long_window_values) / len(long_window_values) if long_window_values else float(default)
			predicted = self.short_weight * short_average + self.long_weight * long_average

			forecast.append(float(predicted))
			series.append(float(predicted))

		return forecast

	def _forecast_binary_series(self, household: Household, key: str, horizon: int, default: float) -> list[float]:
		forecast = self._forecast_series(household, key, horizon, default)
		return [1.0 if value >= self.binary_threshold else 0.0 for value in forecast]