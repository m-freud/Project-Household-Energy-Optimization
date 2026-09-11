from src.runtime_config import RuntimeConfig
from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig
from src.simulation.controllers.mpc.predictors.ml.model_interface import ModelLike
from src.simulation.controllers.mpc.predictors.shared.make_band import make_band
from src.simulation.household import Household
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._pv_gen import _build_pv_gen_features
from src.simulation.controllers.mpc.predictors.ml.global_model.helpers._global_model_prediction import predict_global


def predict_pv_gen(
    model: ModelLike,
    prediction_horizons: list[int],
    household: Household,
    horizon: int,
    interpolation: str = "linear",
    interval_width_frct: float = 0.0,
) -> dict[str, list[float]]:
    if not household.has_pv:
        values = [0.0] * horizon
    else:
        history = list(household.history["pv_gen"].values())
        model_features = [
            *ModelConfig.MODEL_FEATURES_BY_FAMILY["xgboost"]["pv_gen"],
            "prediction_horizon",
        ]
        daylight = RuntimeConfig.PV_GENERATION_WINDOW_ALLOWED

        def build_features(timestep: int, value: float, feature_history: list[float]) -> dict:
            return _build_pv_gen_features(
                timestep, value, feature_history,
                daylight["earliest_start"], daylight["latest_end"],
            )

        values = predict_global(
            model=model,
            prediction_horizons=prediction_horizons,
            current_timestep=household.current_timestep,
            current_value=household.pv_gen,
            history=history,
            horizon=horizon,
            model_features=model_features,
            feature_builder=build_features,
            interpolation=interpolation,
            task="regression",
        )
    lb, ub = make_band(values, interval_width_frct)
    return {"pv_gen": values, "pv_gen_lb": lb, "pv_gen_ub": ub}
