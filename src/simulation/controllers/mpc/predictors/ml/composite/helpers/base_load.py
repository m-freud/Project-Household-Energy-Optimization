from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig
from src.simulation.controllers.mpc.predictors.ml.model_interface import TRegressor
from src.simulation.controllers.mpc.predictors.shared.make_band import make_band
from src.simulation.household import Household
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._base_load import _build_base_load_features
from src.simulation.controllers.mpc.predictors.ml.composite.helpers._composite_prediction import predict_composite

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
