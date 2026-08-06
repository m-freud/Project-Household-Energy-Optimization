from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

import pandas as pd

from src.sqlite_connection import fetch_timeseries, sqlite_cursor


def _fetch_profiles(household_ids: list[int], table_name: str) -> dict:
	profiles = {}

	for household_id in household_ids:
		profile = fetch_timeseries(sqlite_cursor, household_id, table_name)
		profiles[household_id] = profile

	return profiles


def _get_profiles_df(profiles_dict: dict) -> pd.DataFrame:
	rows: list[dict] = []

	for household_id, profile in profiles_dict.items():
		row = {"household_id": household_id}
		for i, value in enumerate(profile):
			row[f"s{i+1}"] = value
		rows.append(row)

	return pd.DataFrame(rows)


def _init_feature_df(profiles_df: pd.DataFrame, value_name: str) -> pd.DataFrame:
	columns = sorted(
		[column for column in profiles_df.columns if str(column).startswith("s")],
		key=lambda name: int(str(name)[1:]),
	)

	if not columns:
		raise ValueError("profiles_df must contain timestep columns s1..sN")

	feature_df = profiles_df.melt(
		id_vars=["household_id"],
		value_vars=columns,
		var_name="timestep_col",
		value_name=value_name,
	)

	feature_df["timestep"] = feature_df["timestep_col"].str[1:].astype(int)
	feature_df["household_id"] = feature_df["household_id"].astype(int)
	feature_df[value_name] = feature_df[value_name].astype(float)

	feature_df = feature_df[["household_id", "timestep", value_name]]
	feature_df = feature_df.sort_values(["household_id", "timestep"]).reset_index(drop=True)

	return feature_df


def _add_running_average_features(
	feature_df: pd.DataFrame,
	windows: tuple[int, ...] = (2, 4, 8, 16),
	value_column: str = "pv_gen",
	prefix: str = "pv",
) -> pd.DataFrame:
	grouped_values = feature_df.groupby("household_id")[value_column]

	for window in windows:
		feature_df[f"{prefix}_ma_{window}"] = (
			grouped_values
			.rolling(window=window, min_periods=1)
			.mean()
			.reset_index(level=0, drop=True)
			.astype(float)
		)

	return feature_df


def _add_std_features(
	feature_df: pd.DataFrame,
	windows: tuple[int, ...] = (4, 8),
	value_column: str = "pv_gen",
	prefix: str = "pv",
) -> pd.DataFrame:
	grouped_values = feature_df.groupby("household_id")[value_column]

	for window in windows:
		feature_df[f"{prefix}_std_{window}"] = (
			grouped_values
			.rolling(window=window, min_periods=1)
			.std(ddof=0)
			.fillna(0.0)
			.reset_index(level=0, drop=True)
			.astype(float)
		)

	return feature_df


def _add_delta_features(
	feature_df: pd.DataFrame,
	value_column: str = "pv_gen",
	prefix: str = "pv",
) -> pd.DataFrame:
	grouped_values = feature_df.groupby("household_id")[value_column]

	prev_1 = grouped_values.shift(1)
	prev_2 = grouped_values.shift(2)

	feature_df[f"{prefix}_delta_1"] = (feature_df[value_column] - prev_1).fillna(0.0).astype(float)
	feature_df[f"{prefix}_delta_2"] = (prev_1 - prev_2).fillna(0.0).astype(float)

	return feature_df


def _add_accel_feature(
	feature_df: pd.DataFrame,
	prefix: str = "pv",
) -> pd.DataFrame:
	delta_1_column = f"{prefix}_delta_1"
	delta_2_column = f"{prefix}_delta_2"
	accel_column = f"{prefix}_accel"

	if delta_1_column not in feature_df.columns or delta_2_column not in feature_df.columns:
		raise ValueError(f"{delta_1_column} and {delta_2_column} are required before computing {accel_column}")

	feature_df[accel_column] = (feature_df[delta_1_column] - feature_df[delta_2_column]).astype(float)
	return feature_df


def _round_float_features(feature_df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
	float_columns = feature_df.select_dtypes(include=["float", "float32", "float64"]).columns
	if len(float_columns) > 0:
		feature_df[float_columns] = feature_df[float_columns].round(int(digits))
	return feature_df
