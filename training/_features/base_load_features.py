from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

import pandas as pd  # noqa: E402

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


def get_base_load_features(household_ids: list[int], round_values: bool = False) -> pd.DataFrame:
	raw_profiles = _fetch_profiles(household_ids, "base_load")
	standardized_df = _get_profiles_df(raw_profiles)
	feature_df = _init_feature_df(standardized_df, target_name="base_load")
	feature_df = _add_trig_time_features(feature_df)
	feature_df = _add_lag_features(
		feature_df,
		source_column="base_load",
		group_cols=("household_id",),
		lags=(1, 2, 4, 8, 12),
		pad_value=-1.0,
		add_pad_flags=True,
		output_prefix="base_load_lag",
		dtype=float,
	)
	feature_df = _add_history_average_features(
		feature_df,
		windows=(2, 4, 8, 16),
		value_column="base_load",
		prefix="base_load",
	)
	feature_df = _add_std_features(
		feature_df,
		windows=(4, 8),
		value_column="base_load",
		prefix="base_load",
	)
	feature_df = _add_delta_features(
		feature_df,
		value_column="base_load",
		prefix="base_load",
	)
	feature_df = _add_accel_feature(feature_df, prefix="base_load")
	feature_df = _add_next_value_target(
		feature_df,
		source_column="base_load",
		group_cols=("household_id",),
		target_column="next_value",
		fill_value=0.0,
		dtype=float,
	)
	if round_values:
		feature_df = _round_float_features(feature_df, digits=3)

	return feature_df


if __name__ == "__main__":
	household_ids = list(range(2, 3))
	feature_df = get_base_load_features(household_ids)

	cols_to_print = [
		"household_id",
		"timestep",
		"base_load",
		"base_load_delta_1",
		"base_load_delta_2",
		"base_load_accel",
		"next_value",
	]

	pd.set_option("display.max_rows", None)
	pd.set_option("display.width", None)

	i = 0
	print(feature_df.iloc[i * 96:(i + 1) * 96][cols_to_print])
