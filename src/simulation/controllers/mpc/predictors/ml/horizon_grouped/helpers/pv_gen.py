from src.runtime_config import RuntimeConfig
from src.simulation.controllers.mpc.predictors.ml.horizon_grouped.helpers.grouped_prediction import (
    get_group_model_features,
    predict_horizon_grouped,
)
from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig
from src.simulation.controllers.mpc.predictors.ml.model_interface import ModelLike
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._pv_gen import (
    _build_pv_gen_features,
    _try_bypass,
)
from src.simulation.controllers.mpc.predictors.shared.make_band import make_band
from src.simulation.household import Household


def predict_pv_gen(
    model_bank: dict[str, ModelLike],
    household: Household,
    horizon: int,
    interval_width_frct: float = 0.0,
) -> dict[str, list[float]]:
    if not household.has_pv:
        values = [0.0] * horizon
    else:
        model_features = get_group_model_features(
            model_bank,
            ModelConfig.MODEL_FEATURES_BY_FAMILY["global_xgboost"]["pv_gen"],
        )
        required_features = model_features[next(iter(model_features))]
        daylight = RuntimeConfig.PV_GENERATION_WINDOW_ALLOWED

        def build_features(timestep: int, value: float, history: list[float]) -> dict:
            return _build_pv_gen_features(
                timestep,
                value,
                history,
                daylight["earliest_start"],
                daylight["latest_end"],
                required_features=required_features,
            )

        def try_bypass(
            timestep: int,
            history: list[float],
            current_value: float,
        ) -> float | None:
            _ = history, current_value
            return _try_bypass(timestep)

        values = predict_horizon_grouped(
            models_by_group=model_bank,
            model_features_by_group=model_features,
            current_timestep=household.current_timestep,
            current_value=household.pv_gen,
            history=list(household.history["pv_gen"].values()),
            horizon=horizon,
            feature_builder=build_features,
            task="regression",
            bypass=try_bypass,
        )

    lower_bound, upper_bound = make_band(values, interval_width_frct)
    return {"pv_gen": values, "pv_gen_lb": lower_bound, "pv_gen_ub": upper_bound}
