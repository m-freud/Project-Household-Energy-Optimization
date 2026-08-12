from pathlib import Path
import sys
import math

import pandas as pd

# enable src imports when running from repo root
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), Path.cwd())
sys.path.insert(0, str(repo_root))

from src.simulation.household import Household
from src.simulation.devices.ev import EV
from src.simulation.controllers.mpc.predictors.xgboost.ev_status import (
    _fetch_ev_status_data,
    _build_ev_status_features,
)
from training._features.ev_status_features import get_ev_status_features


HOUSEHOLD_IDS = [1, 2]
EV_KEYS = ["ev1", "ev2"]
FEATURE_COLUMNS = [
    "timestep",
    "status",
    "time_sin",
    "time_cos",
    "steps_in_current_state",
    "phase_id",
    "status_lag_1",
    "status_lag_1_is_pad",
    "status_lag_2",
    "status_lag_2_is_pad",
    "status_lag_4",
    "status_lag_4_is_pad",
    "status_lag_8",
    "status_lag_8_is_pad",
    "start1_earliest",
    "end1_latest",
    "start2_earliest",
    "end2_latest",
    "max_commute_steps_1",
    "max_commute_steps_2",
    "steps_to_start1_earliest",
    "steps_to_end1_latest",
    "steps_to_start2_earliest",
    "steps_to_end2_latest",
    "start1",
    "end1",
    "start2",
    "end2",
    "start1_observed",
    "end1_observed",
    "start2_observed",
    "end2_observed",
    "observed_window_length_1",
    "observed_window_length_2",
    "window_length_slack_1",
    "window_length_slack_2",
]


def _status_to_home_station(status: int) -> tuple[int, int]:
    if status == 0:
        return 1, 0
    if status == 1:
        return 0, 0
    if status == 2:
        return 0, 1
    raise ValueError(f"Unexpected EV status: {status}")


def _build_household_snapshot(
    profile_df: pd.DataFrame,
    current_timestep: int,
    ev_key: str,
) -> Household:
    ev1 = EV(capacity=1.0, max_charge=1.0, max_discharge=1.0, efficiency=1.0, name="ev1")
    ev2 = EV(capacity=1.0, max_charge=1.0, max_discharge=1.0, efficiency=1.0, name="ev2")
    household = Household(
        player_id=int(profile_df["household_id"].iloc[0]),
        start_time=current_timestep,
        ev1=ev1,
        ev2=ev2,
    )
    household.current_timestep = current_timestep

    # Build history up to t-1 (current timestep state comes from current household attrs).
    home_hist: dict[int, int] = {}
    station_hist: dict[int, int] = {}
    for _, row in profile_df[profile_df["timestep"] < current_timestep].iterrows():
        step = int(row["timestep"])
        at_home, at_station = _status_to_home_station(int(row["status"]))
        home_hist[step] = at_home
        station_hist[step] = at_station

    household.history[f"{ev_key}_at_home"] = home_hist
    household.history[f"{ev_key}_at_charging_station"] = station_hist
    other_ev_key = "ev2" if ev_key == "ev1" else "ev1"
    household.history[f"{other_ev_key}_at_home"] = {}
    household.history[f"{other_ev_key}_at_charging_station"] = {}

    current_status = int(profile_df.loc[profile_df["timestep"] == current_timestep, "status"].iloc[0])
    current_home, current_station = _status_to_home_station(current_status)

    getattr(household, ev_key).at_home = bool(current_home)
    getattr(household, ev_key).at_charging_station = bool(current_station)

    other_ev = household.ev2 if ev_key == "ev1" else household.ev1
    other_ev.at_home = False
    other_ev.at_charging_station = False

    return household


def _equal(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return bool(math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9))
    return a == b


def main() -> None:
    training_df = get_ev_status_features(HOUSEHOLD_IDS)

    mismatch_count = 0
    checked_rows = 0
    mismatch_by_col: dict[str, int] = {}
    mismatch_examples: list[str] = []

    for household_id in HOUSEHOLD_IDS:
        for ev_key in EV_KEYS:
            profile_df = (
                training_df[
                    (training_df["household_id"] == household_id)
                    & (training_df["ev_key"] == ev_key)
                ]
                .sort_values("timestep")
                .reset_index(drop=True)
            )

            for _, train_row in profile_df.iterrows():
                timestep = int(train_row["timestep"])

                household = _build_household_snapshot(
                    profile_df=profile_df,
                    current_timestep=timestep,
                    ev_key=ev_key,
                )

                runtime_data = _fetch_ev_status_data(household, ev_key)
                runtime_features = _build_ev_status_features(runtime_data)

                for col in FEATURE_COLUMNS:
                    runtime_val = runtime_features[col]
                    train_val = train_row[col]
                    if not _equal(runtime_val, train_val):
                        mismatch_count += 1
                        mismatch_by_col[col] = mismatch_by_col.get(col, 0) + 1
                        if len(mismatch_examples) < 25:
                            mismatch_examples.append(
                                f"household={household_id} ev={ev_key} t={timestep} col={col}: "
                                f"runtime={runtime_val} training={train_val}"
                            )

                checked_rows += 1

    print(f"Checked rows: {checked_rows}")
    print(f"Feature mismatches: {mismatch_count}")

    if mismatch_count > 0:
        print("\nFirst mismatches:")
        for example in mismatch_examples:
            print(f"- {example}")

        print("\nMismatch counts by column:")
        for col, count in sorted(mismatch_by_col.items(), key=lambda item: (-item[1], item[0])):
            print(f"- {col}: {count}")

        raise AssertionError(f"Found {mismatch_count} feature mismatches between runtime and training builders")

    print("Runtime and training feature builders match for households 1 and 2.")


if __name__ == "__main__":
    main()
