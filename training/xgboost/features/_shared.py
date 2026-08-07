import numpy as np
import pandas as pd


def _add_trig_time_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    n_timesteps = int(feature_df["timestep"].max())
    angle = 2.0 * np.pi * (feature_df["timestep"].to_numpy() / n_timesteps)
    feature_df["time_sin"] = np.sin(angle)
    feature_df["time_cos"] = np.cos(angle)
    return feature_df


def _add_lag_features(
    feature_df: pd.DataFrame,
    source_column: str,
    group_cols: tuple[str, ...] = ("household_id",),
    lags: tuple[int, ...] = (1, 2, 4, 8),
    pad_value: float | int = -1.0,
    add_pad_flags: bool = True,
    output_prefix: str | None = None,
    dtype: type | None = None,
) -> pd.DataFrame:
    if output_prefix is None:
        output_prefix = f"{source_column}_lag"

    grouped = feature_df.groupby(list(group_cols))[source_column]

    for lag in lags:
        lag_col = f"{output_prefix}_{lag}"
        feature_df[lag_col] = grouped.shift(lag)
        lag_missing = feature_df[lag_col].isna()

        feature_df[lag_col] = feature_df[lag_col].fillna(pad_value)
        if dtype is not None:
            feature_df[lag_col] = feature_df[lag_col].astype(dtype)

        if add_pad_flags:
            feature_df[f"{lag_col}_is_pad"] = lag_missing.astype(int)

    return feature_df


def _add_next_value_target(
    feature_df: pd.DataFrame,
    source_column: str,
    group_cols: tuple[str, ...] = ("household_id",),
    target_column: str | None = None,
    fill_value: float | int = 0.0,
    dtype: type | None = None,
) -> pd.DataFrame:
    if target_column is None:
        target_column = f"next_{source_column}"

    feature_df[target_column] = feature_df.groupby(list(group_cols))[source_column].shift(-1)
    feature_df[target_column] = feature_df[target_column].fillna(fill_value)

    if dtype is not None:
        feature_df[target_column] = feature_df[target_column].astype(dtype)

    return feature_df
