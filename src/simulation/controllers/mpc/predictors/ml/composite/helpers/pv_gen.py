from src.runtime_config import RuntimeConfig
from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig
from src.simulation.controllers.mpc.predictors.ml.model_interface import (
    TRegressor,
    get_model_feature_names,
)
from src.simulation.controllers.mpc.predictors.shared.make_band import make_band
from src.simulation.household import Household
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._pv_gen import (
    _build_pv_gen_features,
    _try_bypass,
)
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
        if model_bank:
            model_features = get_model_feature_names(next(iter(model_bank.values())), model_features)
        daylight = RuntimeConfig.PV_GENERATION_WINDOW_ALLOWED

        def build_features(timestep: int, value: float, feature_history: list[float]) -> dict:
            return _build_pv_gen_features(
                current_timestep=timestep,
                current_pv_gen=value,
                pv_history=feature_history,
                daylight_start=daylight["earliest_start"],
                daylight_end=daylight["latest_end"],
            )

        def try_bypass(
            timestep: int,
            status_history: list[float],
            current_value: float,
        ) -> float | None:
            _ = status_history, current_value
            return _try_bypass(timestep)

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
            bypass=try_bypass,
        )

    pv_gen_lb, pv_gen_ub = make_band(pv_gen, interval_width_frct)
    return {
        "pv_gen": pv_gen,
        "pv_gen_lb": pv_gen_lb,
        "pv_gen_ub": pv_gen_ub,
    }
