# paste this to enable src. imports
from pathlib import Path
import sys

from simulation.household import Household

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))


from simulation.controllers.mpc.predictors.ml.model_interface import TClassifier
from src.simulation.controllers.mpc.predictors.ml.shared_helpers._ev_status import _build_ev_status_features, _try_bypass  # noqa: E402


def _predict_single_ev_status(model_bank, household: Household, ev_key: str, horizon: int) -> tuple[list[int], list[int]]:
    '''Predicts the status of a single EV (at_home, at_charging_station) for the given household and horizon'''
    # Placeholder implementation, replace with actual model prediction logic
    at_home = [1] * horizon
    at_charging_station = [0] * horizon
    return at_home, at_charging_station


def predict_ev_status(
    model_bank_ev1,
    model_bank_ev2,
    household: Household,
    horizon: int,
    ev_key: str | None = None
) -> dict[str, list[int]]:
    '''Predicts ev1 and ev2 status (at_home, at_charging_station) for the given household, horizon'''

    if ev_key in ["ev1", "ev2"]:
        model = model_bank_ev1 if ev_key == "ev1" else model_bank_ev2
        ev_home, ev_station = _predict_single_ev_status(model, household, ev_key, horizon)
        return {
            f"{ev_key}_at_home": ev_home,
                f"{ev_key}_at_charging_station": ev_station
        }
    elif ev_key is not None:
        raise ValueError("wrong ev_key, expected ev1 or ev2")

    ev1_at_home, ev1_at_charging_station = _predict_single_ev_status(model_bank_ev1, household, "ev1", horizon)
    ev2_at_home, ev2_at_charging_station = _predict_single_ev_status(model_bank_ev2, household, "ev2", horizon)

    return {
        "ev1_at_home": ev1_at_home,
        "ev1_at_charging_station": ev1_at_charging_station,
        "ev2_at_home": ev2_at_home,
        "ev2_at_charging_station": ev2_at_charging_station,
    }