from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from xgboost import XGBRegressor
from src.config import Config
from src.sqlite_connection import fetch_timeseries, sqlite_cursor
import pandas as pd
import numpy as np

from training.xgboost.features import _add_trig_time_features


def _fetch_pv_profiles(household_ids: list[int])->dict:
    profiles = {}

    for household_id in household_ids:
        pv_gen = fetch_timeseries(sqlite_cursor, household_id, "pv_gen")
        profiles[household_id] = pv_gen

    return profiles


def _standardize_pv_profiles(raw_profiles: dict) -> pd.DataFrame:
    '''
    returns wide df with columns: household_id, s1..s96
    where s1..s96 are the pv generation values for each timestep
    '''
    pv_profiles_dict = raw_profiles

    rows: list[dict] = []

    for household_id, profile in pv_profiles_dict.items():
        row = {"household_id": household_id}
        for i, value in enumerate(profile):
            row[f"s{i+1}"] = value
        rows.append(row)

    return pd.DataFrame(rows)


def _init_feature_df(profiles_df: pd.DataFrame) -> pd.DataFrame:
    pv_columns = sorted(
        [column for column in profiles_df.columns if str(column).startswith("s")],
        key=lambda name: int(str(name)[1:]),
    )

    if not pv_columns:
        raise ValueError("profiles_df must contain timestep columns s1..sN")

    feature_df = profiles_df.melt(
        id_vars=["household_id"],
        value_vars=pv_columns,
        var_name="timestep_col",
        value_name="pv_gen"
    )

    feature_df["timestep"] = feature_df["timestep_col"].str[1:].astype(int)
    feature_df["household_id"] = feature_df["household_id"].astype(int)
    feature_df["pv_gen"] = feature_df["pv_gen"].astype(float)

    feature_df = feature_df[["household_id", "timestep", "pv_gen"]]
    feature_df = feature_df.sort_values(["household_id", "timestep"]).reset_index(drop=True)

    return feature_df


def _add_pv_lag_features(
    feature_df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 4, 8, 12),
    pad_value: float = -1.0,
    add_pad_flags: bool = True,
) -> pd.DataFrame:
    group_cols = ["household_id"]
    grouped_pv = feature_df.groupby(group_cols)["pv_gen"]

    for lag in lags:
        lag_col = f"pv_lag_{lag}"
        feature_df[lag_col] = grouped_pv.shift(lag)
        lag_missing = feature_df[lag_col].isna()
        feature_df[lag_col] = feature_df[lag_col].fillna(float(pad_value)).astype(float)

        if add_pad_flags:
            feature_df[f"pv_lag_{lag}_is_pad"] = lag_missing.astype(int)

    return feature_df


def _add_pv_running_average_features(
    feature_df: pd.DataFrame,
    windows: tuple[int, ...] = (2, 4, 8, 16),
) -> pd.DataFrame:
    grouped_pv = feature_df.groupby("household_id")["pv_gen"]

    for window in windows:
        feature_df[f"pv_ma_{window}"] = (
            grouped_pv
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .astype(float)
        )

    return feature_df


def _add_pv_std_features(
    feature_df: pd.DataFrame,
    windows: tuple[int, ...] = (4, 8),
) -> pd.DataFrame:
    grouped_pv = feature_df.groupby("household_id")["pv_gen"]

    for window in windows:
        feature_df[f"pv_std_{window}"] = (
            grouped_pv
            .rolling(window=window, min_periods=1)
            .std(ddof=0)
            .fillna(0.0)
            .reset_index(level=0, drop=True)
            .astype(float)
        )

    return feature_df


def _add_pv_delta_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    grouped_pv = feature_df.groupby("household_id")["pv_gen"]

    pv_prev_1 = grouped_pv.shift(1)
    pv_prev_2 = grouped_pv.shift(2)

    feature_df["pv_delta_1"] = (feature_df["pv_gen"] - pv_prev_1).fillna(0.0).astype(float)
    feature_df["pv_delta_2"] = (pv_prev_1 - pv_prev_2).fillna(0.0).astype(float)

    return feature_df


def _add_pv_accel_feature(feature_df: pd.DataFrame) -> pd.DataFrame:
    if "pv_delta_1" not in feature_df.columns or "pv_delta_2" not in feature_df.columns:
        raise ValueError("pv_delta_1 and pv_delta_2 are required before computing pv_accel")

    feature_df["pv_accel"] = (feature_df["pv_delta_1"] - feature_df["pv_delta_2"]).astype(float)
    return feature_df


def _add_steps_to_daylight_boundaries(feature_df: pd.DataFrame) -> pd.DataFrame:
    pv_window = Config.PV_GENERATION_WINDOW_ALLOWED
    
    daylight_start = int(pv_window["earliest_start"])
    daylight_end = int(pv_window["latest_end"])

    timestep = feature_df["timestep"].to_numpy()

    steps_to_start = np.where(timestep <= daylight_start, daylight_start - timestep + 1, daylight_start - timestep)
    steps_to_end = np.where(timestep <= daylight_end, daylight_end - timestep + 1, daylight_end - timestep)

    feature_df["steps_to_daylight_start"] = steps_to_start.astype(int)
    feature_df["steps_to_daylight_end"] = steps_to_end.astype(int)

    return feature_df


def _add_next_pv_gen_target(feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_df["next_pv_gen"] = feature_df.groupby("household_id")["pv_gen"].shift(-1)
    feature_df["next_pv_gen"] = feature_df["next_pv_gen"].fillna(0.0).astype(float)
    return feature_df


def _round_float_features(feature_df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    float_columns = feature_df.select_dtypes(include=["float", "float32", "float64"]).columns
    if len(float_columns) > 0:
        feature_df[float_columns] = feature_df[float_columns].round(int(digits))
    return feature_df



def get_pv_features(household_ids: list[int]) -> pd.DataFrame:
    raw_profiles = _fetch_pv_profiles(household_ids)
    standardized_df = _standardize_pv_profiles(raw_profiles)
    feature_df = _init_feature_df(standardized_df)

    feature_df = _add_trig_time_features(feature_df)
    feature_df = _add_pv_lag_features(feature_df, lags=(1, 2, 4, 8, 12), pad_value=-1.0, add_pad_flags=True)
    feature_df = _add_pv_running_average_features(feature_df, windows=(2, 4, 8, 16))
    feature_df = _add_pv_std_features(feature_df, windows=(4, 8))
    feature_df = _add_pv_delta_features(feature_df)
    feature_df = _add_pv_accel_feature(feature_df)
    feature_df = _add_steps_to_daylight_boundaries(feature_df)
    feature_df = _add_next_pv_gen_target(feature_df)
    feature_df = _round_float_features(feature_df, digits=3)

    return feature_df


if __name__ == "__main__":
    household_ids = list(range(1, 3))
    feature_df = get_pv_features(household_ids)


    cols_to_print = [
    "household_id",
    "timestep",
    "pv_gen",
    # "pv_lag_1",
    # "pv_lag_8",
    # "pv_lag_1_is_pad",
    # "pv_lag_8_is_pad",
    # "pv_ma_2",
    # "pv_ma_8",
    # "pv_std_4",
    # "pv_std_8",
    "pv_delta_1",
    "pv_delta_2",
    "pv_accel",
    "steps_to_daylight_start",
    "steps_to_daylight_end",
    "next_pv_gen",
]

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)

    i = 0
    print(feature_df.iloc[i*96:(i+1)*96][cols_to_print])  # print all timesteps for household i