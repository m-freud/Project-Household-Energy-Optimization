from src.runtime_config import RuntimeConfig
from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig
from src.simulation.controllers.mpc.predictors.ml.model_interface import TRegressor
from src.simulation.controllers.mpc.predictors.shared.make_band import make_band
from src.simulation.household import Household
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._pv_gen import _build_pv_gen_features
from src.simulation.controllers.mpc.predictors.ml.composite.helpers._composite_prediction import predict_composite


def predict_pv_gen(
    model_bank: dict[int, TRegressor],
    household: Household,
    horizon: int,
    interpolation: str = "linear",
    interval_width_frct: float = 0.0,
) -> dict[str, list[float]]:
    if not household.has_pv:
        pv_gen = [0.0] * horizon
    else:
        history = list(household.history["pv_gen"].values())
        model_features = ModelConfig.MODEL_FEATURES_BY_FAMILY["xgboost"]["pv_gen"]
        daylight = RuntimeConfig.PV_GENERATION_WINDOW_ALLOWED

        def build_features(timestep: int, value: float, feature_history: list[float]) -> dict:
            return _build_pv_gen_features(
                current_timestep=timestep,
                current_pv_gen=value,
                pv_history=feature_history,
                daylight_start=daylight["earliest_start"],
                daylight_end=daylight["latest_end"],
            )

        pv_gen = predict_composite(
            model_bank=model_bank,
            current_timestep=household.current_timestep,
            current_value=household.pv_gen,
            history=history,
            horizon=horizon,
            model_features=model_features,
            feature_builder=build_features,
            interpolation=interpolation,
            task="regression",
        )

    pv_gen_lb, pv_gen_ub = make_band(pv_gen, interval_width_frct)
    return {
        "pv_gen": pv_gen,
        "pv_gen_lb": pv_gen_lb,
        "pv_gen_ub": pv_gen_ub,
    }
