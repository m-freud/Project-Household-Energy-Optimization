

from collections.abc import Callable
from functools import partial
from pathlib import Path
from src.simulation.controllers.base_controller import BaseController
from src.simulation.controllers.mpc.config.device_buffer_config import DeviceBufferConfig
from src.simulation.controllers.mpc.mpc_controller import MPCController
from src.simulation.controllers.mpc.predictors.base_predictor import BasePredictor
from src.simulation.controllers.mpc.predictors.history_avg.history_avg_predictor import HistoryAveragePredictor
from src.simulation.controllers.mpc.predictors.ml.composite.composite_ml_predictor import CompositeMLPredictor
from src.simulation.controllers.mpc.predictors.ml.global_model.global_model_ml_predictor import GlobalModelMLPredictor
from src.simulation.controllers.mpc.predictors.ml.recursive.recursive_ml_predictor import RecursiveMLPredictor
from src.simulation.controllers.mpc.predictors.oracle.oracle_predictor import OraclePredictor
from src.simulation.controllers.stepwise.step_functions.basic_examples import no_control
from src.simulation.controllers.stepwise.step_functions.linear.linear import even_linear_policy, fast_charge_policy
from src.simulation.controllers.stepwise.step_functions.linear.price_aware_linear import price_aware_linear
from src.simulation.controllers.stepwise.step_functions.waterfall.waterfall import waterfall_policy
from src.simulation.controllers.stepwise.stepwise_controller import StepwiseController
from src.simulation.household import Household
from src.simulation.scenarios.scenario import Scenario
from src.runtime_config import RuntimeConfig
from typing import Callable, TypeVar
from xgboost import XGBClassifier, XGBRegressor
import re

TXGBModel = TypeVar("TXGBModel", XGBRegressor, XGBClassifier)
TLoadedModel = TypeVar("TLoadedModel")


def build_function_controller(
    household: Household,
    scenario: Scenario,
    *,
    name: str,
    step_function: Callable[..., dict],
) -> StepwiseController:
    _ = (household, scenario)
    return StepwiseController(name=name, step_function=step_function)


def make_function_controller( # for parallel runs
    name: str,
    step_function: Callable[..., dict],
) -> Callable[[Household, Scenario], StepwiseController]:
    return partial(build_function_controller, name=name, step_function=step_function)


def build_mpc_controller(
    household: Household,
    scenario: Scenario,
    *,
    name: str,
    horizon: int = 96,
    predictor: BasePredictor,
    duration_hours: float = 0.25,
    buffer_config: DeviceBufferConfig | None = None,
) -> MPCController:
    return MPCController(
        name=name,
        household=household,
        scenario=scenario,
        horizon=horizon,
        predictor=predictor,
        duration_hours=duration_hours,
        buffer_config=buffer_config,
    )


def make_mpc_controller(
    name: str,
    horizon: int=96,
    predictor: BasePredictor=None,
    duration_hours: float = 0.25,
    buffer_config: DeviceBufferConfig | None = None,
) -> Callable[[Household, Scenario], MPCController]:
    return partial(
        build_mpc_controller,
        name=name,
        horizon=horizon,
        predictor=predictor,
        duration_hours=duration_hours,
        buffer_config=buffer_config,
    )



def _load_xgb_model(model_path: Path, model_class: type[TXGBModel]) -> TXGBModel:
    model = model_class()
    try:
        model.load_model(Path(model_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load XGBoost model from '{model_path}'.") from exc
    return model


def _single_model_path(family: str, target: str) -> Path:
    family_key = str(family).lower()
    family_config = RuntimeConfig.MODEL_FAMILY_CONFIGS[family_key]
    model_dir = family_config.target_model_dirs[target]
    suffix = family_config.file_suffix

    preferred = Path(model_dir / f"model{suffix}")
    if preferred.exists():
        return preferred

    fallback = Path(model_dir / f"fold_1{suffix}")
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Missing single model for {family_key}:{target}. Looked for {preferred} and {fallback}."
    )

def _load_composite_model_bank(
    target: str,
    model_class: type[TXGBModel],
    model_directories_by_target: dict[str, Path],
) -> dict[int, TXGBModel]:
    model_dir = model_directories_by_target[target] / "per_horizon_single_step"
    model_paths = sorted(model_dir.glob("single_y*.json"))
    model_bank: dict[int, TXGBModel] = {}
    for model_path in model_paths:
        match = re.fullmatch(r"single_y(\d+)\.json", model_path.name)
        if match is not None:
            model_bank[int(match.group(1))] = _load_xgb_model(model_path, model_class)
    if not model_bank:
        raise FileNotFoundError(
            f"No per-horizon models found for '{target}' in '{model_dir}'."
        )
    return model_bank


def build_single_step_predictor(
    model_directories_by_target: dict[str, Path],
) -> RecursiveMLPredictor:
    return RecursiveMLPredictor(
        base_load_model=_load_xgb_model(
            model_directories_by_target["base_load"] / "single_step" / "single_step.json",
            XGBRegressor,
        ),
        pv_gen_model=_load_xgb_model(
            model_directories_by_target["pv_gen"] / "single_step" / "single_step.json",
            XGBRegressor,
        ),
        ev1_status_model=_load_xgb_model(
            model_directories_by_target["ev1_status"] / "single_step" / "single_step.json",
            XGBClassifier,
        ),
        ev2_status_model=_load_xgb_model(
            model_directories_by_target["ev2_status"] / "single_step" / "single_step.json",
            XGBClassifier,
        ),
    )


def build_multi_step_predictor(
    model_directories_by_target: dict[str, Path],
) -> GlobalModelMLPredictor:
    return GlobalModelMLPredictor(
        base_load_model=_load_xgb_model(
            model_directories_by_target["base_load"] / "multi_step" / "multi_step.json",
            XGBRegressor,
        ),
        pv_gen_model=_load_xgb_model(
            model_directories_by_target["pv_gen"] / "multi_step" / "multi_step.json",
            XGBRegressor,
        ),
        ev1_model=_load_xgb_model(
            model_directories_by_target["ev1_status"] / "multi_step" / "multi_step.json",
            XGBClassifier,
        ),
        ev2_model=_load_xgb_model(
            model_directories_by_target["ev2_status"] / "multi_step" / "multi_step.json",
            XGBClassifier,
        ),
    )


def build_composite_horizons_predictor(
    model_directories_by_target: dict[str, Path],
) -> CompositeMLPredictor:
    return CompositeMLPredictor(
        base_load_model_bank=_load_composite_model_bank(
            "base_load",
            XGBRegressor,
            model_directories_by_target,
        ),
        pv_gen_model_bank=_load_composite_model_bank(
            "pv_gen",
            XGBRegressor,
            model_directories_by_target,
        ),
        ev1_model_bank=_load_composite_model_bank(
            "ev1_status",
            XGBClassifier,
            model_directories_by_target,
        ),
        ev2_model_bank=_load_composite_model_bank(
            "ev2_status",
            XGBClassifier,
            model_directories_by_target,
        ),
    )


def build_controller_factories(
    requested_names: list[str] | None = None,
) -> dict[str, Callable[[Household, Scenario], BaseController]]:
    available_names = {
        "no_control",
        "fast_charge",
        "even_linear",
        "price_aware_linear",
        "waterfall",
        "mpc_oracle",
        "mpc_history_avg",
        "mpc_single_step",
        "mpc_multi_step",
        "mpc_composite_horizons",
    }
    selected_names = available_names if requested_names is None else set(requested_names)
    unknown_names = selected_names - available_names
    if unknown_names:
        raise ValueError(f"Unknown controllers: {sorted(unknown_names)}")

    factories: dict[str, Callable[[Household, Scenario], BaseController]] = {}
    function_controllers = {
        "no_control": no_control,
        "fast_charge": fast_charge_policy,
        "even_linear": even_linear_policy,
        "price_aware_linear": partial(
            price_aware_linear,
            base_behaviour="even_linear",
        ),
        "waterfall": waterfall_policy,
    }
    for name, step_function in function_controllers.items():
        if name in selected_names:
            factories[name] = make_function_controller(name, step_function)

    if "mpc_oracle" in selected_names:
        factories["mpc_oracle"] = make_mpc_controller(
            "mpc_oracle",
            horizon=96,
            predictor=OraclePredictor(),
        )
    if "mpc_history_avg" in selected_names:
        factories["mpc_history_avg"] = make_mpc_controller(
            "mpc_history_avg",
            predictor=HistoryAveragePredictor(conf_interval_frct=0.0),
        )

    if not any(name in selected_names for name in {"mpc_single_step", "mpc_multi_step", "mpc_composite_horizons"}):
        return factories

    model_directories_by_target = {
        "base_load": RuntimeConfig.ROOT_DIR / "models" / "load",
        "pv_gen": RuntimeConfig.ROOT_DIR / "models" / "pv",
        "ev1_status": RuntimeConfig.ROOT_DIR / "models" / "ev1",
        "ev2_status": RuntimeConfig.ROOT_DIR / "models" / "ev2",
    }
    predictor_builders = {
        "mpc_single_step": build_single_step_predictor,
        "mpc_multi_step": build_multi_step_predictor,
        "mpc_composite_horizons": build_composite_horizons_predictor,
    }
    for name, build_predictor in predictor_builders.items():
        if name in selected_names:
            factories[name] = make_mpc_controller(
                name,
                horizon=96,
                predictor=build_predictor(model_directories_by_target),
            )

    return factories