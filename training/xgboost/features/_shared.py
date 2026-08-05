import numpy as np
import pandas as pd

def _add_trig_time_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    n_timesteps = int(feature_df["timestep"].max())
    angle = 2.0 * np.pi * (feature_df["timestep"].to_numpy() / n_timesteps)
    feature_df["time_sin"] = np.sin(angle)
    feature_df["time_cos"] = np.cos(angle)
    return feature_df
