
# paste this to enable src. imports
import itertools
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

import argparse  # noqa
import json  # noqa
import matplotlib.pyplot as plt  # noqa
import pandas as pd  # noqa
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa
from sklearn.linear_model import Ridge, RidgeClassifier  # noqa
from sklearn.metrics import root_mean_squared_error  # noqa
from sklearn.pipeline import Pipeline  # noqa
from sklearn.preprocessing import StandardScaler  # noqa
from xgboost import XGBClassifier, XGBRegressor  # noqa

from src.simulation.controllers.mpc.predictors.ml.model_config import MODEL_FEATURES_BY_FAMILY  # noqa
from training.features.base_load_features import get_base_load_features  # noqa
from training.features.ev_status_features import get_ev_status_features  # noqa
from training.features.pv_gen_features import get_pv_gen_features  # noqa
from training.split.clean_split import PARTITIONS  # noqa

from training.tuning.pred_hypam_sweep import (  # noqa
    GRID_MAP,
    build_model,
    build_param_grid,
    get_train_test_frames,
    normalize_model,
    normalize_target,
)
from training.training_ch_load.sampling.sampling import ALL_IDS, DIAG_FEATURE_DIR  # noqa

from src.simulation.controllers.mpc.predictors.ml.model_config import MODEL_FEATURES_BY_FAMILY  # noqa

OUTPUT_DIR = Path(__file__).parent

xgb_best_params = {"learning_rate": 0.03, "max_depth": 2, "n_estimators": 200} # best params from load sweep on original data

def load_feature_sample_df(start_id=100354, n_days=365, model_family="xgboost"):
    features = MODEL_FEATURES_BY_FAMILY[model_family]["base_load"]
    columns = features + ["next_value"]

    start_idx = ALL_IDS.index(str(start_id))
    days_needed = n_days
    day_frames = []

    for file_id in ALL_IDS[start_idx:]:
        if days_needed <= 0:
            break

        file_df = pd.read_parquet(DIAG_FEATURE_DIR / f"diag_{file_id}_features.parquet")
        n_available_days = len(file_df) // 96
        n_take_days = min(days_needed, n_available_days)

        day_frames.append(file_df.iloc[: n_take_days * 96][columns])
        days_needed -= n_take_days

    if days_needed > 0:
        raise ValueError(f"not enough diag feature samples starting from {start_id}: missing {days_needed} days")

    return pd.concat(day_frames, ignore_index=True)


def sweep_n_days(
    model_family="xgboost",
    train_start_id=100354,
    test_sample_ids=[107448, 108517],
    max_n_days=1460,
    plateau_tol=1e-3,
):
    features = MODEL_FEATURES_BY_FAMILY[model_family]["base_load"]
    params = xgb_best_params if model_family == "xgboost" else {}

    test_df = pd.concat(
        [load_feature_sample_df(start_id=test_id, n_days=365, model_family=model_family) for test_id in test_sample_ids],
        ignore_index=True,
    )
    X_test, y_test = test_df[features], test_df["next_value"]

    rows = []
    prev_rmse = None
    for n_days in range(1, max_n_days + 1, 1):
        train_df = load_feature_sample_df(start_id=train_start_id, n_days=n_days, model_family=model_family)

        model = build_model(model_family, "base_load", params)
        model.fit(train_df[features], train_df["next_value"])
        rmse = root_mean_squared_error(y_test, model.predict(X_test))

        print(f"n_train_days={n_days} rmse={rmse:.4f}")
        rows.append({"n_train_days": n_days, "rmse": rmse})

        # stop early once additional training days stop improving rmse
        if prev_rmse is not None and abs(prev_rmse - rmse) < plateau_tol:
            print(f"rmse plateaued at n_train_days={n_days}, stopping sweep")
            continue # break

        prev_rmse = rmse

    result_df = pd.DataFrame(rows)

    output_dir = Path(__file__).parent / "train_size_sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_dir / f"{model_family}_train_size_sweep.csv", index=False)

    fig, ax = plt.subplots()
    ax.plot(result_df["n_train_days"], result_df["rmse"], marker="o")
    ax.set_xlabel("n_train_days")
    ax.set_ylabel("rmse")
    ax.set_title(f"{model_family} base_load train size sweep")
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_family}_train_size_sweep.png")
    plt.close(fig)

    return result_df




def hypam_sweep(n_years, model, grid):
    pass

if __name__ == "__main__":
    sweep_n_days(max_n_days=120)
    exit(0)


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["xgboost"],
        help="inference model(s). choose between xgboost, random_forest, ridge",
    )
    parser.add_argument(
        "--grid",
        type=str,
        nargs="+",
        default=["grid_1"],
        help="grid(s)",
    )
    args = parser.parse_args()

    models = [normalize_model(m) for m in args.model]
    grids = args.grid

    for model in models:
        for grid in grids:
            if grid not in GRID_MAP.get(model, {}):
                print(f"Skipping {model}/{grid}: grid not defined for model")
                continue
            hypam_sweep(n_years=1, model=model, grid=grid)

