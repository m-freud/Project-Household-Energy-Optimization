from __future__ import annotations

from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.simulation.household import Household
from src.simulation.scenarios.scenario import Scenario


class MovingAveragePredictor(BasePredictor):
	"""Predict future profiles using a rolling average over recent history.

	When the simulation has only a short history so far, the predictor warms up
	from the current observed value. This avoids the
	start-of-day case where a pure trailing average would otherwise be too flat.
	"""

	def __init__(self, window_size: int = 12, binary_threshold: float = 0.5):
		self.window_size = max(1, int(window_size))
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
		# Causal warm-up: never read future/oracle profiles for padding.
		return [float(default)] * horizon

	def _seed_series(self, household: Household, key: str, default: float) -> list[float]:
		history = self._history_values(household, key)
		seed = history[-self.window_size :]

		if len(seed) >= self.window_size:
			return seed

		needed = self.window_size - len(seed)
		if seed:
			# Keep warm-up causal while preserving the earliest observed level.
			return [float(seed[0])] * needed + seed

		fallback = self._fallback_series(household, key, self.window_size, default)
		return fallback[:needed]

	def _forecast_series(self, household: Household, key: str, horizon: int, default: float) -> list[float]:
		if horizon <= 0:
			return []

		series = self._seed_series(household, key, default)
		forecast: list[float] = []

		for _ in range(horizon):
			window = series[-self.window_size :]
			predicted = sum(window) / len(window) if window else float(default)
			forecast.append(float(predicted))
			series.append(float(predicted))

		return forecast

	def _forecast_binary_series(self, household: Household, key: str, horizon: int, default: float) -> list[float]:
		forecast = self._forecast_series(household, key, horizon, default)
		return [1.0 if value >= self.binary_threshold else 0.0 for value in forecast]
