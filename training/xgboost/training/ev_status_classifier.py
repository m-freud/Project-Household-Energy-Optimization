# EV status classifier for xgboost
# we want to train a classifier that yields the position of an EV for the next n timesteps
# n = remaining timesteps
# the mpc solver needs length 96 but we can pad the tail

# the classifier only predicts the next step, then starts again from there until the horizon is reached.
# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from xgboost import XGBClassifier
from src.config import Config
from src.sqlite_connection import fetch_timeseries, sqlite_cursor
from training.xgboost.features.ev_status_features import get_ev_status_features
from training.split import distinct_set_strict, distinct_set_ignore_ev_status

import pandas as pd
import numpy as np

household_ids = distinct_set_ignore_ev_status

print(f"Household IDs for training: {household_ids}")

feature_df = get_ev_status_features(household_ids)

# X_train, y_train =

# model.fit

# test model

# save model