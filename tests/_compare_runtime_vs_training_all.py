from pathlib import Path
import math
import sys

import pandas as pd

# enable src imports when running from repo root
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), Path.cwd())
sys.path.insert(0, str(repo_root))

from src.simulation.household import Household
from src.simulation.devices.ev import EV
from src.simulation.devices.pv import PV
from src.sqlite_connection import fetch_timeseries, sqlite_cursor

from src.simulation.controllers.mpc.predictors.xgboost.pv_gen import (
    _fetch_pv_gen_profiles,
    _build_pv_gen_features,
    MODEL_FEATURE_COLUMNS as PV_MODEL_FEATURE_COLUMNS,
)
from src.simulation.controllers.mpc.predictors.xgboost.ev_status import (
    _fetch_ev_status_data,
    _build_ev_status_features,
    MODEL_FEATURE_COLUMNS as EV_MODEL_FEATURE_COLUMNS,
)
from src.simulation.controllers.mpc.predictors.xgboost.base_load import (
    _build_base_load_features,
    MODEL_FEATURE_COLUMNS as BASE_LOAD_MODEL_FEATURE_COLUMNS,
)
from training.xgboost.features.pv_gen_features import get_pv_features
from training.xgboost.features.ev_status_features import get_ev_status_features
from training.xgboost.features.base_load_features import get_base_load_features


HOUSEHOLD_IDS = [1, 2]
EV_KEYS = ["ev1", "ev2"]


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


def _current_status(home_series: list[float], station_series: list[float], current_timestep: int) -> int:
    idx = int(current_timestep) - 1
    return int(1 - int(home_series[idx]) + int(station_series[idx]))


def _build_household_snapshot(
    household_id: int,
    current_timestep: int,
    pv_series: list[float],
    base_load_series: list[float],
    ev1_home_series: list[float],
    ev1_station_series: list[float],
    ev2_home_series: list[float],
    ev2_station_series: list[float],
) -> Household:
    current_idx = int(current_timestep) - 1

    ev1_home, ev1_station = _status_to_home_station(_current_status(ev1_home_series, ev1_station_series, current_timestep))
    ev2_home, ev2_station = _status_to_home_station(_current_status(ev2_home_series, ev2_station_series, current_timestep))

    household = Household(
        player_id=int(household_id),
        start_time=int(current_timestep),
        pv=PV(generation=float(pv_series[current_idx])),
        ev1=EV(capacity=1.0, max_charge=1.0, max_discharge=1.0, efficiency=1.0, at_home=bool(ev1_home), at_charging_station=bool(ev1_station), name="ev1"),
        ev2=EV(capacity=1.0, max_charge=1.0, max_discharge=1.0, efficiency=1.0, at_home=bool(ev2_home), at_charging_station=bool(ev2_station), name="ev2"),
    )
    household.current_timestep = int(current_timestep)
    household.base_load = float(base_load_series[current_idx])

    household.history["pv_gen"] = _series_to_history(pv_series, current_timestep)
    household.history["base_load"] = _series_to_history(base_load_series, current_timestep)
    household.history["ev1_at_home"] = _series_to_history(ev1_home_series, current_timestep)
    household.history["ev1_at_charging_station"] = _series_to_history(ev1_station_series, current_timestep)
    household.history["ev2_at_home"] = _series_to_history(ev2_home_series, current_timestep)
    household.history["ev2_at_charging_station"] = _series_to_history(ev2_station_series, current_timestep)

    return household


def _equal(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return bool(math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9))
    return a == b


def _compare_metric(
    metric_name: str,
    training_df: pd.DataFrame,
    model_columns: list[str],
    runtime_feature_fn,
    household_ids: list[int],
    fetch_training_row_fn,
) -> None:
    mismatch_count = 0
    checked_rows = 0
    mismatch_by_col: dict[str, int] = {}
    mismatch_examples: list[str] = []

    for household_id in household_ids:
        profile_df = fetch_training_row_fn(training_df, household_id)

        for _, train_row in profile_df.iterrows():
            current_timestep = int(train_row["timestep"])
            household = _build_household_snapshot(
                household_id=household_id,
                current_timestep=current_timestep,
                pv_series=list(fetch_timeseries(sqlite_cursor, household_id, "pv_gen")),
                base_load_series=list(fetch_timeseries(sqlite_cursor, household_id, "base_load")),
                ev1_home_series=list(fetch_timeseries(sqlite_cursor, household_id, "ev1_at_home")),
                ev1_station_series=list(fetch_timeseries(sqlite_cursor, household_id, "ev1_at_charging_station")),
                ev2_home_series=list(fetch_timeseries(sqlite_cursor, household_id, "ev2_at_home")),
                ev2_station_series=list(fetch_timeseries(sqlite_cursor, household_id, "ev2_at_charging_station")),
            )

            runtime_features = runtime_feature_fn(household, current_timestep, train_row)

            for col in model_columns:
                runtime_val = runtime_features[col]
                train_val = train_row[col]
                if not _equal(runtime_val, train_val):
                    mismatch_count += 1
                    mismatch_by_col[col] = mismatch_by_col.get(col, 0) + 1
                    if len(mismatch_examples) < 25:
                        mismatch_examples.append(
                            f"household={household_id} t={current_timestep} col={col}: runtime={runtime_val} training={train_val}"
                        )

            checked_rows += 1

    print(f"[{metric_name}] Checked rows: {checked_rows}")
    print(f"[{metric_name}] Feature mismatches: {mismatch_count}")

    if mismatch_count > 0:
        print(f"\n[{metric_name}] First mismatches:")
        for example in mismatch_examples:
            print(f"- {example}")

        print(f"\n[{metric_name}] Mismatch counts by column:")
        for col, count in sorted(mismatch_by_col.items(), key=lambda item: (-item[1], item[0])):
            print(f"- {col}: {count}")

        raise AssertionError(f"[{metric_name}] Found {mismatch_count} feature mismatches")


def main() -> None:
    pv_training = get_pv_features(HOUSEHOLD_IDS)
    ev_training = get_ev_status_features(HOUSEHOLD_IDS)
    base_load_training = get_base_load_features(HOUSEHOLD_IDS)

    def _pv_rows(df: pd.DataFrame, household_id: int) -> pd.DataFrame:
        return df[df["household_id"] == household_id].sort_values("timestep").reset_index(drop=True)

    def _ev_rows(df: pd.DataFrame, household_id: int) -> pd.DataFrame:
        return df[df["household_id"] == household_id].sort_values(["ev_key", "timestep"]).reset_index(drop=True)

    def _base_rows(df: pd.DataFrame, household_id: int) -> pd.DataFrame:
        return df[df["household_id"] == household_id].sort_values("timestep").reset_index(drop=True)

    _compare_metric(
        metric_name="pv_gen",
        training_df=pv_training,
        model_columns=PV_MODEL_FEATURE_COLUMNS,
        runtime_feature_fn=lambda household, current_timestep, train_row: _build_pv_gen_features(
            _fetch_pv_gen_profiles(
                household=household,
                current_timestep=current_timestep,
                current_pv_gen=float(household.pv_gen),
                pv_history=household.history.get("pv_gen", {}),
            )
        ),
        household_ids=HOUSEHOLD_IDS,
        fetch_training_row_fn=_pv_rows,
    )

    _compare_metric(
        metric_name="ev_status",
        training_df=ev_training,
        model_columns=EV_MODEL_FEATURE_COLUMNS,
        runtime_feature_fn=lambda household, current_timestep, train_row: _build_ev_status_features(
            _fetch_ev_status_data(household, str(train_row["ev_key"]))
        ),
        household_ids=HOUSEHOLD_IDS,
        fetch_training_row_fn=lambda df, household_id: _ev_rows(df, household_id),
    )

    _compare_metric(
        metric_name="base_load",
        training_df=base_load_training,
        model_columns=BASE_LOAD_MODEL_FEATURE_COLUMNS,
        runtime_feature_fn=lambda household, current_timestep, train_row: _build_base_load_features(
            household=household,
            current_timestep=current_timestep,
            current_base_load=float(household.base_load),
            base_load_history=[float(value) for _, value in sorted(household.history.get("base_load", {}).items())],
        ),
        household_ids=HOUSEHOLD_IDS,
        fetch_training_row_fn=_base_rows,
    )

    print("All three runtime feature builders match training for households 1 and 2.")


if __name__ == "__main__":
    main()