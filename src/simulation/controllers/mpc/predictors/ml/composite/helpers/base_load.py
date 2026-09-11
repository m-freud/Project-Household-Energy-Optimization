from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig
from src.simulation.controllers.mpc.predictors.ml.model_interface import TRegressor
from src.simulation.controllers.mpc.predictors.shared.make_band import make_band
from src.simulation.household import Household
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._base_load import _build_base_load_features
from src.simulation.controllers.mpc.predictors.ml.composite.helpers._composite_prediction import predict_composite
from matplotlib import pyplot as plt
from typing import Any, cast

def predict_base_load(
    model_bank: dict[int, TRegressor],
    household: Household,
    horizon: int,
    interpolation: str = "linear",
    interval_width_frct: float = 0.0,
) -> dict[str, list[float]]:
    history = list(household.history["base_load"].values())
    model_features = ModelConfig.MODEL_FEATURES_BY_FAMILY["xgboost"]["base_load"]

    def build_features(timestep: int, value: float, feature_history: list[float]) -> dict:
        return _build_base_load_features(
            current_timestep=timestep,
            current_base_load=value,
            base_load_history=feature_history,
        )

    base_load = predict_composite(
        model_bank=model_bank,
        current_timestep=household.current_timestep,
        current_value=household.base_load,
        history=history,
        horizon=horizon,
        model_features=model_features,
        feature_builder=build_features,
        interpolation=interpolation,
        task="regression",
    )
    base_load_lb, base_load_ub = make_band(base_load, interval_width_frct)
    return {
        "base_load": base_load,
        "base_load_lb": base_load_lb,
        "base_load_ub": base_load_ub,
    }


if __name__ == "__main__":
    class _DebugConstantRegressor:
        def __init__(self, value: float):
            self.value = value

        def predict(self, model_input):
            return [self.value] * len(model_input) # * random.choice([-1, 0, 1])] * len(model_input)


    def _debug_run_base_load_interpolation() -> None:
        debug_household = Household(player_id=1, start_time=1)
        debug_household.current_timestep = 20
        debug_household.base_load = 10.0
        debug_household.history["base_load"] = {timestep: 10.0 for timestep in range(1, 20)}

        debug_model_bank = cast(
            Any,
            {horizon: _DebugConstantRegressor(horizon) for horizon in (1, 2, 4, 8, 16, 32, 64)},
        )

        for interpolation in ("linear", "recursive"):
            prediction_horizon = 96
            base_load_history = list(debug_household.history["base_load"].values())
            all_features = _build_base_load_features(
                current_timestep=debug_household.current_timestep,
                current_base_load=debug_household.base_load,
                base_load_history=base_load_history,
            )
            model_features = ModelConfig.MODEL_FEATURES_BY_FAMILY["xgboost"]["base_load"]
            initial_predictions = {0: debug_household.base_load}
            for horizon, model in debug_model_bank.items():
                model_input = [all_features[feature] for feature in model_features]
                initial_predictions[horizon] = float(model.predict([model_input])[0])

            prediction_dict = dict(initial_predictions)
            prediction = _interpolate_prediction(
                model_bank=debug_model_bank,
                history=base_load_history,
                pred_dict=prediction_dict,
                horizon=prediction_horizon,
                current_timestep=debug_household.current_timestep,
                interpolation=interpolation,
            )
            interpolated_horizons = sorted(set(prediction_dict) - set(initial_predictions))

            plt.figure(figsize=(12, 5))
            plt.plot(range(prediction_horizon), prediction, color="red", linewidth=1, label="full prediction")
            initial_horizons = sorted(initial_predictions)
            plt.scatter(
                initial_horizons,
                [initial_predictions[horizon] for horizon in initial_horizons],
                color="blue",
                zorder=3,
                label="initial model predictions",
            )
            plt.scatter(
                interpolated_horizons,
                [prediction_dict[horizon] for horizon in interpolated_horizons],
                color="red",
                zorder=3,
                label="interpolated / recursive predictions",
            )
            plt.title(f"Base-load composite prediction: {interpolation}")
            plt.xlabel("Forecast horizon")
            plt.ylabel("Base load")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

    _debug_run_base_load_interpolation()