# EV status classifier for xgboost
# we want to train a classifier that yields the position of an EV for the next n timesteps
# n = remaining timesteps
# the mpc solver needs length 96 but we can pad the tail

# the classifier only predicts the next step, then starts again from there until the horizon is reached.
# paste this to enable src. imports
import math
from pathlib import Path
import re
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from xgboost import XGBClassifier
from src.config import Config
from src.sqlite_connection import fetch_timeseries, sqlite_cursor
import pandas as pd
import numpy as np


def _fetch_raw_ev_status_profiles(household_ids: list[int])->dict:
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


def _init_feature_df(profiles_df: pd.DataFrame) -> pd.DataFrame:
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


def _encode_time_cyclic(timestep: int, total_timesteps: int) -> tuple[float, float]:
    """
    Encode a timestep as cyclic features (sine and cosine).

    Args:
        timestep (int): The current timestep to encode.
        total_timesteps (int): The total number of timesteps in the cycle.

    Returns:
        tuple[float, float]: A tuple containing the sine and cosine values for the encoded timestep.
    """
    angle = 2 * math.pi * (timestep / total_timesteps)
    return math.sin(angle), math.cos(angle)


def _add_trig_time_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    n_timesteps = int(feature_df["timestep"].max())
    angle = 2.0 * np.pi * (feature_df["timestep"].to_numpy() / n_timesteps)
    feature_df["time_sin"] = np.sin(angle)
    feature_df["time_cos"] = np.cos(angle)
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

    phase_to_id = {
        "home1": 0,
        "drive1": 1,
        "station": 2,
        "drive2": 3,
        "home2": 4,
    }

    def _phase_for_group(group: pd.DataFrame) -> pd.DataFrame:
        statuses = group["status"].astype(int).tolist()
        phases: list[str] = []
        seen_station = False

        for idx, current_status in enumerate(statuses):
            _ = idx
            if current_status == 2:
                phase = "station"
                seen_station = True
            elif current_status == 1:
                phase = "drive2" if seen_station else "drive1"
            elif current_status == 0:
                phase = "home2" if seen_station else "home1"
            else:
                phase = "home2" if seen_station else "home1"

            phases.append(phase)

        group = group.copy()
        group["phase"] = phases
        group["phase_id"] = group["phase"].map(phase_to_id).astype(int)
        return group

    feature_df = feature_df.groupby(group_cols, group_keys=False).apply(_phase_for_group)
    return feature_df


def _add_status_lag_features(
    feature_df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 4, 8),
    pad_value: int = -1,
    add_pad_flags: bool = True,
) -> pd.DataFrame:
    group_cols = ["household_id", "ev_key"]
    grouped_status = feature_df.groupby(group_cols)["status"]

    for lag in lags:
        lag_col = f"status_lag_{lag}"
        feature_df[lag_col] = grouped_status.shift(lag)
        lag_missing = feature_df[lag_col].isna()
        feature_df[lag_col] = feature_df[lag_col].fillna(pad_value).astype(int)

        if add_pad_flags:
            feature_df[f"status_lag_{lag}_is_pad"] = lag_missing.astype(int)

    return feature_df


def _add_allowed_commute_window_boundaries(feature_df: pd.DataFrame) -> pd.DataFrame:
    windows = Config.EV_COMMUTE_WINDOWS_ALLOWED

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
    windows = Config.EV_COMMUTE_WINDOWS_ALLOWED

    feature_df["max_commute_steps_1"] = feature_df["ev_key"].map(
        {"ev1": int(windows["ev1"][0]["max_unavailable_steps"]), "ev2": int(windows["ev2"][0]["max_unavailable_steps"])}
    ).astype(int)
    feature_df["max_commute_steps_2"] = feature_df["ev_key"].map(
        {"ev1": int(windows["ev1"][1]["max_unavailable_steps"]), "ev2": int(windows["ev2"][1]["max_unavailable_steps"])}
    ).astype(int)
    return feature_df


def _add_times_to_boundary(feature_df: pd.DataFrame) -> pd.DataFrame:
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

    def _add_group_boundaries(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("timestep").copy()

        drive1_timesteps = group.loc[group["phase_id"] == 1, "timestep"]
        drive2_timesteps = group.loc[group["phase_id"] == 3, "timestep"]

        start1_time = int(drive1_timesteps.iloc[0]) if not drive1_timesteps.empty else -1
        end1_time = int(drive1_timesteps.iloc[-1]) if not drive1_timesteps.empty else -1
        start2_time = int(drive2_timesteps.iloc[0]) if not drive2_timesteps.empty else -1
        end2_time = int(drive2_timesteps.iloc[-1]) if not drive2_timesteps.empty else -1

        phase_id = group["phase_id"].astype(int)

        # Default unknown (-1). Unlock each observed boundary only after phase progression.
        group["start1"] = np.where(phase_id > 0, start1_time, -1)
        group["end1"] = np.where(phase_id > 1, end1_time, -1)
        group["start2"] = np.where(phase_id > 2, start2_time, -1)
        group["end2"] = np.where(phase_id > 3, end2_time, -1)

        return group

    feature_df = feature_df.groupby(group_cols, group_keys=False).apply(_add_group_boundaries)
    feature_df[["start1", "end1", "start2", "end2"]] = feature_df[["start1", "end1", "start2", "end2"]].astype(int)
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


def _add_next_state(feature_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["household_id", "ev_key"]
    feature_df["next_state"] = feature_df.groupby(group_cols)["status"].shift(-1)
    feature_df["next_state"] = feature_df["next_state"].fillna(0).astype(int)
    return feature_df





# we train with H 101-250
household_ids = list(range(101, 251))

# get raw data from sqlite:  {id: {ev1_at_home: [0,1,..], ev1_at_station: [0,1,..], ev2_at_home: [0,1,..], ev2_at_station: [0,1,..]}}
raw_profiles = _fetch_raw_ev_status_profiles(household_ids)

# standardize to df with 1 row per series: 
# household | ev | s1 | s2 | s3 | ... | s96
# 101       | ev1| 0  | 1  | 0  | ... | 1
# 101       | ev2| 1  | 0  | 0  | ... | 0


status_profiles_df = _standardize_status_profiles(raw_profiles=raw_profiles)
# print(f"Fetched {len(status_profiles_df)} EV rows.")
# print(status_profiles_df.head(3))

# get features and labels for training -> one row per H-EV-timestep
# household_id ev_key  timestep  status
# 101          ev1      1          0
# ...

# init
feature_df = _init_feature_df(status_profiles_df)
print(f"Feature rows: {len(feature_df)}")

# ADD FEATURES!

# trig time features
feature_df = _add_trig_time_features(feature_df)

# current state duration
feature_df = _add_steps_in_current_state(feature_df)

# phase feature
feature_df = _add_phase_feature(feature_df)

# lag features from status history
feature_df = _add_status_lag_features(feature_df, lags=(1, 2, 3, 4, 8, 12))

# allowed commute-window constraints (config priors)
feature_df = _add_allowed_commute_window_boundaries(feature_df)

# signed non-zero time to each allowed boundary
feature_df = _add_times_to_boundary(feature_df)

# # max unavailable steps per commute window
feature_df = _add_max_commute_steps(feature_df)

# # observed commute-window boundaries from realized profile (or -1 if absent)
feature_df = _add_observed_commute_window_boundaries(feature_df)

# # binary flags for observed boundaries
feature_df = _add_observed_boundary_flags(feature_df)

# observed commute-window lengths (else -1)
feature_df = _add_observed_window_length(feature_df)

# slack to configured max commute lengths (else -1)
feature_df = _add_window_length_slack(feature_df)

# target for one-step prediction; final row defaults to 0
feature_df = _add_next_state(feature_df)


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


# X_train, y_train =

# model.fit

# test model

# save model