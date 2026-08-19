from __future__ import annotations

from collections.abc import Mapping

from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.simulation.household import Household


class ModularPredictor(BasePredictor):
    """Compose multiple predictors by routing selected targets to specific models.

    The default predictor produces a complete prediction dictionary. Any entry in
    ``target_predictors`` overrides that key with the value produced by the mapped
    predictor. This keeps the class compatible with all existing predictor classes
    as long as they implement ``predict(household, horizon)``.
    """
    #TODO dont do full predictions and then overwrite. instead do targeted predictions. this works for now but its slow

    def __init__(
        self,
        *,
        default_predictor: BasePredictor,
        target_predictors: Mapping[str, BasePredictor] | None = None,
    ):
        self.default_predictor = default_predictor
        self.target_predictors = dict(target_predictors or {})
        self.targets = ["base_load", "pv_gen", "ev1_status", "ev2_status"]

    def predict(self, household: Household, horizon: int) -> dict:
        prediction = dict(self.default_predictor.predict(household, horizon))

        for target_name, predictor in self.target_predictors.items():
            target_prediction = predictor.predict(household, horizon)
            if target_name not in target_prediction:
                available = ", ".join(sorted(target_prediction.keys()))
                raise KeyError(
                    f"Predictor for target '{target_name}' did not return that key. "
                    f"Available keys: {available}"
                )
            prediction[target_name] = target_prediction[target_name]

        return prediction