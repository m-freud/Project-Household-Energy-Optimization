import unittest
from typing import Any

import numpy as np

from src.simulation.controllers.mpc.predictors.xgboost.base_load import predict_base_load
from src.simulation.household import Household
from src.simulation.devices.ev import EV


class ArrayOnlyStubModel:
    def __init__(self):
        self.calls = []

    def predict(self, X: Any):
        assert isinstance(X, np.ndarray)
        self.calls.append(np.asarray(X).shape)
        return np.asarray([1.0] * len(X), dtype=float)


class PredictBaseLoadTest(unittest.TestCase):
    def test_predict_base_load_accepts_numpy_feature_matrix(self):
        ev1 = EV(capacity=1.0, max_charge=1.0, max_discharge=1.0, efficiency=1.0, name="ev1")
        ev2 = EV(capacity=1.0, max_charge=1.0, max_discharge=1.0, efficiency=1.0, name="ev2")
        household = Household(player_id=1, start_time=1, ev1=ev1, ev2=ev2)
        household.base_load = 0.5
        household.current_timestep = 1

        model = ArrayOnlyStubModel()
        result = predict_base_load(model, household, horizon=3)

        self.assertEqual(result["base_load"], [0.5, 1.0, 1.0])
        self.assertEqual(result["base_load_lb"], [0.5, 1.0, 1.0])
        self.assertEqual(result["base_load_ub"], [0.5, 1.0, 1.0])
        self.assertEqual(len(model.calls), 2)


if __name__ == "__main__":
    unittest.main()
