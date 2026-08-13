from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.config import Config  # noqa: E402
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

from training._features._regression import (  # noqa: E402
    _fetch_profiles,
    _get_profiles_df,
    _init_feature_df,
    _add_history_average_features,
    _add_std_features,
    _add_delta_features,
    _add_accel_feature,
    _round_float_features,
)
from training._features._shared import (  # noqa: E402
    _add_trig_time_features,
    _add_lag_features,
    _add_next_value_target,
)


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



def get_pv_gen_features(household_ids: list[int], round_values: bool = False) -> pd.DataFrame:
    raw_profiles = _fetch_profiles(household_ids, "pv_gen")
    standardized_df = _get_profiles_df(raw_profiles)
    feature_df = _init_feature_df(standardized_df, value_name="pv_gen")

    feature_df = _add_trig_time_features(feature_df)
    feature_df = _add_lag_features(
        feature_df,
        source_column="pv_gen",
        group_cols=("household_id",),
        lags=(1, 2, 4, 8, 12),
        pad_value=-1.0,
        add_pad_flags=True,
        output_prefix="pv_lag",
        dtype=float,
    )
    feature_df = _add_history_average_features(feature_df, windows=(2, 4, 8, 16))
    feature_df = _add_std_features(feature_df, windows=(4, 8), value_column="pv_gen", prefix="pv")
    feature_df = _add_delta_features(feature_df, value_column="pv_gen", prefix="pv")
    feature_df = _add_accel_feature(feature_df, prefix="pv")
    feature_df = _add_steps_to_daylight_boundaries(feature_df)
    feature_df = _add_next_value_target(
        feature_df,
        source_column="pv_gen",
        group_cols=("household_id",),
        target_column="next_value",
        fill_value=0.0,
        dtype=float,
    )
    
    if round_values:
        feature_df = _round_float_features(feature_df, digits=3)
    
    return feature_df


if __name__ == "__main__":
    household_ids = list(range(1, 3))
    feature_df = get_pv_gen_features(household_ids)


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
    "next_value",
]

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)

    i = 0
    print(feature_df.iloc[i*96:(i+1)*96][cols_to_print])  # print all timesteps for household i