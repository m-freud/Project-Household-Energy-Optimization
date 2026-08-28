import xgboost as xgb
import sqlite3
import pandas as pd
# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))


from training._features.base_load_features import get_base_load_features

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
from training._features._shared import ( # noqa: E402
    _add_trig_time_features, 
	_add_lag_features, 
    _add_next_value_target
  )  

sqlite_path = "sqlite/ch_smart_meters.db"

with sqlite3.connect(sqlite_path) as conn:
    load_df = pd.read_sql("SELECT * FROM load", conn)

# only use full days
load_df = load_df[
    load_df["timestamp_utc"].between(
        "2023-01-01 00:00:00",
        "2024-12-30 23:45:00"
    )
]

n_days = len(load_df) // 96
split_idx = int(n_days * 0.8) * 96

train_df = load_df.iloc[:split_idx]
val_df = load_df.iloc[split_idx:]

def make_feature_df(load_df: pd.DataFrame, round_values: bool = True) -> pd.DataFrame:
    print("lets make feature df")
    feature_df = pd.DataFrame()

    for date, day_df in load_df.groupby(
        pd.to_datetime(load_df["timestamp_utc"]).dt.date
        ):
        for player_id in day_df.columns[2:]:
            player_day_df = pd.DataFrame({
                "timestep": day_df["period"].to_numpy(),
                "household_id": int(player_id),
                "load": day_df[player_id].to_numpy(),
            })

            player_day_df = _add_trig_time_features(player_day_df)
            player_day_df = _add_lag_features(
                player_day_df,
                source_column="load",
                group_cols=("household_id",),
                lags=(1, 2, 4, 8, 12),
                pad_value=-1.0,
                add_pad_flags = False,
                output_prefix="base_load_lag",
                dtype=float,
            )
            player_day_df = _add_history_average_features(
                player_day_df,
                windows=(2, 4, 8, 16),
                value_column="load",
                prefix="base_load",
            )
            player_day_df = _add_std_features(
                player_day_df,
                windows=(4, 8),
                value_column="load",
                prefix="base_load",
            )
            player_day_df = _add_delta_features(
                player_day_df,
                value_column="load",
                prefix="base_load",
            )
            player_day_df = _add_accel_feature(
                player_day_df,
                prefix="base_load",
            )
            player_day_df = _add_next_value_target(
                player_day_df,
                source_column="load",
                group_cols=("household_id",),
                target_column="next_value",
                fill_value=0.0,
                dtype=float,
            )
            if round_values:
                player_day_df = _round_float_features(player_day_df, digits=3)

            feature_df = pd.concat([feature_df, player_day_df], ignore_index=True)

    return feature_df



train_feature_df = make_feature_df(train_df)

print(f"Train feature df: {len(train_feature_df)} rows, {len(train_feature_df.columns)} columns")
print(train_feature_df.head(10))

exit()




def create_feature_df(): pass
    # cycle thorugh all players / days and create features for each timestep




profile_1_df = fetch_player_load_profile(100354)

print(f"Profile 1: {len(profile_1_df)} rows, {len(profile_1_df.columns) - 2} columns")
print(profile_1_df.head(10))



# i want X_train, y_train, X_val, y_val
# so i need full raw profiles -> get features -> feature df