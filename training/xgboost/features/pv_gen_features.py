from xgboost import XGBRegressor
from src.config import Config
from src.sqlite_connection import fetch_timeseries, sqlite_cursor
import pandas as pd
import numpy as np

from training.xgboost.features import _add_trig_time_features


def _fetch_pv_profiles(household_ids: list[int])->dict:
    profiles = {}

    for household_id in household_ids:
        pv_gen = fetch_timeseries(sqlite_cursor, "pv_gen", household_id)
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



def get_pv_features(household_ids: list[int]) -> pd.DataFrame:
    raw_profiles = _fetch_pv_profiles(household_ids)
    standardized_df = _standardize_pv_profiles(raw_profiles)
    feature_df = _init_feature_df(standardized_df)

    feature_df = _add_trig_time_features(feature_df)


    return feature_df