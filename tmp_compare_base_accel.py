import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from training.xgboost.features.base_load_features import get_base_load_features
from src.sqlite_connection import sqlite_cursor, fetch_timeseries
from src.simulation.controllers.mpc.predictors.xgboost.base_load import _build_base_load_features
from tests._compare_runtime_vs_training_features import _build_household_for_base_load


df = get_base_load_features([1])
rows = df[df['household_id'] == 1].sort_values('timestep').reset_index(drop=True)
for timestep in [4, 5, 7, 9, 10, 11, 14, 18, 22, 25, 28, 29, 30, 31, 33]:
    row = rows.loc[rows['timestep'] == timestep].iloc[0]
    household = _build_household_for_base_load(1, int(timestep), fetch_timeseries(sqlite_cursor, 1, 'base_load'))
    runtime = _build_base_load_features(
        current_timestep=int(timestep),
        current_base_load=float(row['base_load']),
        base_load_history=[float(v) for _, v in sorted(household.history.get('base_load', {}).items())],
        n_evs_at_home=int(row['n_evs_at_home']),
        round_values=True,
    )
    print('timestep', timestep)
    print('training', row['base_load_accel'], 'delta1', row['base_load_delta_1'], 'delta2', row['base_load_delta_2'])
    print('runtime', runtime['base_load_accel'], 'delta1', runtime['base_load_delta_1'], 'delta2', runtime['base_load_delta_2'])
    print()
