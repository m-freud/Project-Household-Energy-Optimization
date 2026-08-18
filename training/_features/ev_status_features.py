import math
from pathlib import Path
import re
import pandas as pd
import numpy as np
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.runtime_config import RuntimeConfig  # noqa: E402
from src.sqlite_connection import fetch_timeseries, sqlite_cursor  # noqa: E402


from training._features._shared import _add_trig_time_features, _add_lag_features, _add_next_value_target  # noqa: E402


def _fetch_ev_status_profiles(household_ids: list[int])->dict:
    data = {}
    for household_id in household_ids:
        ev1_at_home = fetch_timeseries(sqlite_cursor, table_name="ev1_at_home", player_id=household_id)
        ev1_at_station = fetch_timeseries(sqlite_cursor, table_name="ev1_at_charging_station", player_id=household_id)
        ev2_at_home = fetch_timeseries(sqlite_cursor, table_name="ev2_at_home", player_id=household_id)
        ev2_at_station = fetch_timeseries(sqlite_cursor, table_name="ev2_at_charging_station", player_id=household_id)

        # keep raw for this step
        data[household_id] = {
            "ev1_at_home": ev1_at_home,
            "ev1_at_station": ev1_at_station,
            "ev2_at_home": ev2_at_home,
            "ev2_at_station": ev2_at_station,
        }

    return data


def _standardize_status_profiles(raw_profiles: dict) -> pd.DataFrame:
    '''
    returns wide df with columns: household_id, ev_key, s1..s96
    where s1..s96 are the status values for each timestep
    encoded as 0=at_home, 1=commuting, 2=at_station
    '''
    raw_data = raw_profiles

    status_profiles_dict = {}

    for household_id in raw_data.keys():
        ev1_at_home = raw_data[household_id]["ev1_at_home"]
        ev1_at_station = raw_data[household_id]["ev1_at_station"]
        ev2_at_home = raw_data[household_id]["ev2_at_home"]
        ev2_at_station = raw_data[household_id]["ev2_at_station"]

        # 0 = at home, 1 = commuting, 2 = at charging station
        # ev status = 1 - at_home + at_station
        ev1_status = [1 - at_home + at_station for at_home, at_station in zip(ev1_at_home, ev1_at_station)]
        ev2_status = [1 - at_home + at_station for at_home, at_station in zip(ev2_at_home, ev2_at_station)]

        status_profiles_dict[household_id] = {
            "ev1_status": ev1_status,
            "ev2_status": ev2_status,
        }

    rows: list[dict] = []
    for household_id, ev_statuses in status_profiles_dict.items():
        for ev_key in ("ev1", "ev2"):
            series = ev_statuses[f"{ev_key}_status"]
            row = {
                "household_id": int(household_id),
                "ev_key": ev_key,
            }
            for idx, state in enumerate(series, start=1):
                row[f"s{idx}"] = int(state)
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["household_id", "ev_key"]).reset_index(drop=True)
    return df


def _init_status_feature_df(profiles_df: pd.DataFrame) -> pd.DataFrame:
    status_columns = sorted(
        [column for column in profiles_df.columns if re.fullmatch(r"s\d+", str(column))],
        key=lambda name: int(name[1:]),
    )

    if not status_columns:
        raise ValueError("profiles_df must contain status columns s1..s96")

    feature_df = profiles_df.melt(
        id_vars=["household_id", "ev_key"],
        value_vars=status_columns,
        var_name="timestep_col",
        value_name="status",
    )

    feature_df["timestep"] = feature_df["timestep_col"].str[1:].astype(int)
    feature_df["status"] = feature_df["status"].astype(int)

    feature_df = feature_df[["household_id", "ev_key", "timestep", "status"]]
    feature_df = feature_df.sort_values(["household_id", "ev_key", "timestep"]).reset_index(drop=True)
    return feature_df


def _add_steps_in_current_state(feature_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["household_id", "ev_key"]

    previous_status = feature_df.groupby(group_cols)["status"].shift(1)
    state_changed = (feature_df["status"] != previous_status).astype(int)
    segment_id = state_changed.groupby([feature_df[col] for col in group_cols]).cumsum()

    feature_df["steps_in_current_state"] = (
        feature_df.groupby(group_cols + [segment_id]).cumcount() + 1
    ).astype(int)

    return feature_df


def _add_phase_feature(feature_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["household_id", "ev_key"]

    # Per profile, mark if we already reached station (status==2) at or before this timestep.
    seen_station = (
        feature_df["status"].eq(2)
        .groupby([feature_df[col] for col in group_cols])
        .cummax()
    )

    phase = np.select(
        [
            feature_df["status"].eq(2),
            feature_df["status"].eq(1) & (~seen_station),
            feature_df["status"].eq(1) & seen_station,
            feature_df["status"].eq(0) & (~seen_station),
        ],
        ["station", "drive1", "drive2", "home1"],
        default="home2",
    )

    phase_to_id = {
        "home1": 0,
        "drive1": 1,
        "station": 2,
        "drive2": 3,
        "home2": 4,
    }

    feature_df["phase"] = phase
    feature_df["phase_id"] = pd.Series(phase, index=feature_df.index).map(phase_to_id).astype(int)
    return feature_df


def _add_allowed_commute_window_boundaries(feature_df: pd.DataFrame) -> pd.DataFrame:
    windows = RuntimeConfig.EV_COMMUTE_WINDOWS_ALLOWED

    feature_df["start1_earliest"] = feature_df["ev_key"].map(
        {"ev1": int(windows["ev1"][0]["earliest_start"]), "ev2": int(windows["ev2"][0]["earliest_start"])}
    ).astype(int)
    feature_df["end1_latest"] = feature_df["ev_key"].map(
        {"ev1": int(windows["ev1"][0]["latest_end"]), "ev2": int(windows["ev2"][0]["latest_end"])}
    ).astype(int)
    feature_df["start2_earliest"] = feature_df["ev_key"].map(
        {"ev1": int(windows["ev1"][1]["earliest_start"]), "ev2": int(windows["ev2"][1]["earliest_start"])}
    ).astype(int)
    feature_df["end2_latest"] = feature_df["ev_key"].map(
        {"ev1": int(windows["ev1"][1]["latest_end"]), "ev2": int(windows["ev2"][1]["latest_end"])}
    ).astype(int)
    return feature_df


def _add_max_commute_steps(feature_df: pd.DataFrame) -> pd.DataFrame:
    windows = RuntimeConfig.EV_COMMUTE_WINDOWS_ALLOWED

    feature_df["max_commute_steps_1"] = feature_df["ev_key"].map(
        {"ev1": int(windows["ev1"][0]["max_unavailable_steps"]), "ev2": int(windows["ev2"][0]["max_unavailable_steps"])}
    ).astype(int)
    feature_df["max_commute_steps_2"] = feature_df["ev_key"].map(
        {"ev1": int(windows["ev1"][1]["max_unavailable_steps"]), "ev2": int(windows["ev2"][1]["max_unavailable_steps"])}
    ).astype(int)
    return feature_df


def _add_steps_to_boundary_feature(feature_df: pd.DataFrame) -> pd.DataFrame:
    boundary_columns = [
        "start1_earliest",
        "end1_latest",
        "start2_earliest",
        "end2_latest",
    ]

    missing = [column for column in boundary_columns if column not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing boundary columns for time-to-boundary features: {missing}")

    timestep = feature_df["timestep"].to_numpy()
    for boundary_col in boundary_columns:
        boundary = feature_df[boundary_col].to_numpy()
        # Keep strictly non-zero values around the boundary: ... 2, 1, -1, -2, ...
        values = np.where(timestep <= boundary, boundary - timestep + 1, boundary - timestep)
        feature_df[f"steps_to_{boundary_col}"] = values.astype(int)

    return feature_df


def _add_observed_commute_window_boundaries(feature_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["household_id", "ev_key"]

    if "phase_id" not in feature_df.columns:
        raise ValueError("phase_id is required. Call _add_phase_feature before observed boundaries.")

    grouping_keys = [feature_df[col] for col in group_cols]

    t_drive1 = feature_df["timestep"].where(feature_df["phase_id"] == 1)
    t_drive2 = feature_df["timestep"].where(feature_df["phase_id"] == 3)

    start1_time = t_drive1.groupby(grouping_keys).transform("min")
    end1_time = t_drive1.groupby(grouping_keys).transform("max")
    start2_time = t_drive2.groupby(grouping_keys).transform("min")
    end2_time = t_drive2.groupby(grouping_keys).transform("max")

    start1_time = start1_time.fillna(-1).astype(int)
    end1_time = end1_time.fillna(-1).astype(int)
    start2_time = start2_time.fillna(-1).astype(int)
    end2_time = end2_time.fillna(-1).astype(int)

    phase_id = feature_df["phase_id"].astype(int)

    feature_df["start1"] = np.where(phase_id > 0, start1_time, -1).astype(int)
    feature_df["end1"] = np.where(phase_id > 1, end1_time, -1).astype(int)
    feature_df["start2"] = np.where(phase_id > 2, start2_time, -1).astype(int)
    feature_df["end2"] = np.where(phase_id > 3, end2_time, -1).astype(int)

    return feature_df


def _add_observed_boundary_flags(feature_df: pd.DataFrame) -> pd.DataFrame:
    for boundary in ("start1", "end1", "start2", "end2"):
        feature_df[f"{boundary}_observed"] = (feature_df[boundary] != -1).astype(int)
    return feature_df


def _add_observed_window_length(feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_df["observed_window_length_1"] = np.where(
        feature_df["end1_observed"] == 1,
        feature_df["end1"] - feature_df["start1"] + 1,
        -1,
    ).astype(int)

    feature_df["observed_window_length_2"] = np.where(
        feature_df["end2_observed"] == 1,
        feature_df["end2"] - feature_df["start2"] + 1,
        -1,
    ).astype(int)

    return feature_df


def _add_window_length_slack(feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_df["window_length_slack_1"] = np.where(
        feature_df["observed_window_length_1"] != -1,
        feature_df["max_commute_steps_1"] - feature_df["observed_window_length_1"],
        -1,
    ).astype(int)

    feature_df["window_length_slack_2"] = np.where(
        feature_df["observed_window_length_2"] != -1,
        feature_df["max_commute_steps_2"] - feature_df["observed_window_length_2"],
        -1,
    ).astype(int)

    return feature_df


def get_ev_status_features(household_ids: list[int]) -> pd.DataFrame:
    raw_profiles = _fetch_ev_status_profiles(household_ids)
    profiles_df = _standardize_status_profiles(raw_profiles)
    feature_df = _init_status_feature_df(profiles_df)

    feature_df = _add_trig_time_features(feature_df)
    feature_df = _add_steps_in_current_state(feature_df)
    feature_df = _add_phase_feature(feature_df)
    feature_df = _add_lag_features(
        feature_df,
        source_column="status",
        group_cols=("household_id", "ev_key"),
        lags=(1, 2, 4, 8),
        pad_value=-1,
        add_pad_flags=True,
        output_prefix="status_lag",
        dtype=int,
    )
    feature_df = _add_allowed_commute_window_boundaries(feature_df)
    feature_df = _add_max_commute_steps(feature_df)
    feature_df = _add_steps_to_boundary_feature(feature_df)
    feature_df = _add_observed_commute_window_boundaries(feature_df)
    feature_df = _add_observed_boundary_flags(feature_df)
    feature_df = _add_observed_window_length(feature_df)
    feature_df = _add_window_length_slack(feature_df)
    feature_df = _add_next_value_target(
        feature_df,
        source_column="status",
        group_cols=("household_id", "ev_key"),
        target_column="next_state",
        fill_value=0,
        dtype=int,
    )

    return feature_df


if __name__ == "__main__":
    household_ids = list(range(101, 251))
    feature_df = get_ev_status_features(household_ids)
    print(feature_df.head())


    # pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)

    cols_to_print = [
    "household_id",
    "ev_key",
    "timestep",
    "status",
    "next_state",
    "start1_earliest",
    # "end1_latest",
    # "start2_earliest",
    # "end2_latest",
    # "max_commute_steps_1",
    # "max_commute_steps_2",
    "start1",
    # "end1",
    # "start2",
    # "end2",
    "start1_observed",
    # "end1_observed",
    # "start2_observed",
    # "end2_observed",
    "observed_window_length_1",
    # "observed_window_length_2",
    # "window_length_slack_1",
    # "window_length_slack_2",
    "steps_to_start1_earliest",
    # "steps_to_end1_latest",
    # "steps_to_start2_earliest",
    # "steps_to_end2_latest",
    # "steps_in_current_state",
    "phase",
    ] # + [f"status_lag_{lag}" for lag in (2, 4, 8)] + [f"status_lag_{lag}_is_pad" for lag in (2, 4, 8)]

    i = 1
    print(feature_df.iloc[i*96:(i+2)*96][cols_to_print])  # print all timesteps for household i
