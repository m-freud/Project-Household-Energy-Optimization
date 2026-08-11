from pathlib import Path
import math
import sys
from numbers import Number

import pandas as pd
import numpy as np

# enable src imports when running from repo root
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), Path.cwd())
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.simulation.household import Household
from src.simulation.devices.ev import EV
from src.simulation.devices.pv import PV
from src.sqlite_connection import fetch_timeseries, sqlite_cursor

from src.simulation.controllers.mpc.predictors.xgboost.base_load import (
    _build_base_load_features,
)
from src.simulation.controllers.mpc.predictors.xgboost.pv_gen import (
    _build_pv_gen_features,
)
from src.simulation.controllers.mpc.predictors.xgboost.ev_status import (
    _fetch_ev_status_data,
    _build_ev_status_features,
)

from training.xgboost.features.base_load_features import get_base_load_features
from training.xgboost.features.pv_gen_features import get_pv_gen_features
from training.xgboost.features.ev_status_features import get_ev_status_features

HOUSEHOLD_IDS = [1, 6]


def _series_to_history(series: list[float], current_timestep: int) -> dict[int, float]:
    return {step: float(series[step - 1]) for step in range(1, int(current_timestep))}


def _status_to_home_station(status: int) -> tuple[int, int]:
    if status == 0:
        return 1, 0
    if status == 1:
        return 0, 0
    if status == 2:
        return 0, 1
    raise ValueError(f"Unexpected EV status: {status}")


def _build_household_for_base_load(household_id: int, timestep: int, base_load_series: list[float]) -> Household:
    household = Household(player_id=int(household_id), start_time=int(timestep))
    household.current_timestep = int(timestep)
    household.base_load = float(base_load_series[timestep - 1])
    household.history["base_load"] = _series_to_history(base_load_series, timestep)
    return household


def _build_household_for_pv(household_id: int, timestep: int, pv_series: list[float]) -> Household:
    household = Household(player_id=int(household_id), start_time=int(timestep), pv=PV(generation=float(pv_series[timestep - 1])))
    household.current_timestep = int(timestep)
    household.history["pv_gen"] = _series_to_history(pv_series, timestep)
    return household


def _build_household_for_ev_status(
    household_id: int,
    timestep: int,
    ev1_home_series: list[float],
    ev1_station_series: list[float],
    ev2_home_series: list[float],
    ev2_station_series: list[float],
) -> Household:
    current_idx = int(timestep) - 1
    ev1_home, ev1_station = _status_to_home_station(int(1 - int(ev1_home_series[current_idx]) + int(ev1_station_series[current_idx])))
    ev2_home, ev2_station = _status_to_home_station(int(1 - int(ev2_home_series[current_idx]) + int(ev2_station_series[current_idx])))

    household = Household(
        player_id=int(household_id),
        start_time=int(timestep),
        ev1=EV(capacity=1.0, max_charge=1.0, max_discharge=1.0, efficiency=1.0, at_home=bool(ev1_home), at_charging_station=bool(ev1_station), name="ev1"),
        ev2=EV(capacity=1.0, max_charge=1.0, max_discharge=1.0, efficiency=1.0, at_home=bool(ev2_home), at_charging_station=bool(ev2_station), name="ev2"),
    )
    household.current_timestep = int(timestep)
    household.history["ev1_at_home"] = _series_to_history(ev1_home_series, timestep)
    household.history["ev1_at_charging_station"] = _series_to_history(ev1_station_series, timestep)
    household.history["ev2_at_home"] = _series_to_history(ev2_home_series, timestep)
    household.history["ev2_at_charging_station"] = _series_to_history(ev2_station_series, timestep)
    return household


def _values_equal(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, Number) and isinstance(b, Number):
        return math.isclose(float(a), float(b), rel_tol=1e-3, abs_tol=1e-3)
    return a == b


def _compare_feature_rows(
    label: str,
    training_df: pd.DataFrame,
    feature_columns: list[str],
    build_runtime_features,
    build_household,
    household_ids: list[int],
) -> None:
    mismatches: list[str] = []
    checked = 0

    for household_id in household_ids:
        household_rows = training_df[training_df["household_id"] == household_id].sort_values("timestep").reset_index(drop=True)
        for _, row in household_rows.iterrows():
            timestep = int(row["timestep"])
            household = build_household(household_id, timestep, row)
            runtime_features = build_runtime_features(household, timestep, row)
            for column in feature_columns:
                runtime_value = runtime_features.get(column)
                training_value = row.get(column)
                if not _values_equal(runtime_value, training_value):
                    mismatches.append(
                        f"{label}: household={household_id}, timestep={timestep}, column={column}: runtime={runtime_value!r}, training={training_value!r}"
                    )
            checked += 1

    if mismatches:
        print(f"[{label}] mismatches found: {len(mismatches)}")
        for mismatch in mismatches[:20]:
            print(f"- {mismatch}")
    else:
        print(f"[{label}] checked {checked} rows; parity OK")


def main() -> None:
    base_load_training = get_base_load_features(HOUSEHOLD_IDS)
    pv_training = get_pv_gen_features(HOUSEHOLD_IDS)
    ev_training = get_ev_status_features(HOUSEHOLD_IDS)

    _compare_feature_rows(
        label="base_load",
        training_df=base_load_training,
        feature_columns=Config.XGB_FEATURES["BASE_LOAD"],
        build_runtime_features=lambda household, timestep, row: _build_base_load_features(
            current_timestep=int(timestep),
            current_base_load=float(row["base_load"]),
            base_load_history=[float(value) for _, value in sorted(household.history.get("base_load", {}).items())],
            n_evs_at_home=int(row["n_evs_at_home"]),
            round_values=True,
        ),
        build_household=lambda household_id, timestep, row: _build_household_for_base_load(
            household_id=household_id,
            timestep=timestep,
            base_load_series=fetch_timeseries(sqlite_cursor, household_id, "base_load"),
        ),
        household_ids=HOUSEHOLD_IDS,
    )

    _compare_feature_rows(
        label="pv_gen",
        training_df=pv_training,
        feature_columns=Config.XGB_FEATURES["PV_GEN"],
        build_runtime_features=lambda household, timestep, row: _build_pv_gen_features(
            current_timestep=int(timestep),
            current_pv_gen=float(row["pv_gen"]),
            pv_history=[float(value) for _, value in sorted(household.history.get("pv_gen", {}).items())],
            daylight_start=int(Config.PV_GENERATION_WINDOW_ALLOWED["earliest_start"]),
            daylight_end=int(Config.PV_GENERATION_WINDOW_ALLOWED["latest_end"]),
            round_values=True,
        ),
        build_household=lambda household_id, timestep, row: _build_household_for_pv(
            household_id=household_id,
            timestep=timestep,
            pv_series=fetch_timeseries(sqlite_cursor, household_id, "pv_gen"),
        ),
        household_ids=HOUSEHOLD_IDS,
    )

    _compare_feature_rows(
        label="ev_status",
        training_df=ev_training,
        feature_columns=Config.XGB_FEATURES["EV_STATUS"],
        build_runtime_features=lambda household, timestep, row: _build_ev_status_features(
            _fetch_ev_status_data(household, str(row["ev_key"]))
        ),
        build_household=lambda household_id, timestep, row: _build_household_for_ev_status(
            household_id=household_id,
            timestep=timestep,
            ev1_home_series=fetch_timeseries(sqlite_cursor, household_id, "ev1_at_home"),
            ev1_station_series=fetch_timeseries(sqlite_cursor, household_id, "ev1_at_charging_station"),
            ev2_home_series=fetch_timeseries(sqlite_cursor, household_id, "ev2_at_home"),
            ev2_station_series=fetch_timeseries(sqlite_cursor, household_id, "ev2_at_charging_station"),
        ),
        household_ids=HOUSEHOLD_IDS,
    )


if __name__ == "__main__":
    main()
