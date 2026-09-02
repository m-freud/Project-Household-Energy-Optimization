
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

OUTPUT_DIR = Path(__file__).parent

xgb_best_params = {"learning_rate": 0.03, "max_depth": 2, "n_estimators": 200} # best params from load sweep on original data

ALL_IDS = ['100354', '100707', '102016', '103995', '104481', '105366', '106298', '106793', '107448', '108517', '108795', '109970', '110260', '111381','112377', '113377', '115937', '116320', '118984', '119559', '120517', '120691', '121500', '121555', '122670', '123129', '125463', '127225', '127524', '128843', '129071', '130253', '131514', '132256', '132967', '133165', '133996', '136407', '137077', '138414', '139410', '140211', '140649', '142975', '142999', '144492', '146624', '147049', '147147', '151683', '151740', '152601', '155209', '157904', '159093', '160216', '160431', '161573', '162680', '163329', '163595', '163677', '163817', '164897', '165786', '166084', '166206', '167494', '168165', '169310', '170013', '171530', '172262', '172510', '174968', '176249', '176350', '179723', '180146', '180931', '183144', '184973', '185196', '187422', '188093', '188131', '188208', '189416', '191628', '191701', '192815', '198650', '199888']



def load_feature_sample_df(start_id=100354, n_days=120, model_family="xgboost", date_steps=3):
    features = MODEL_FEATURES_BY_FAMILY[model_family]["base_load"]
    columns = features + ["next_value"]
    sample_path = DIAG_FEATURE_DIR / f"diag_{start_id}_features.parquet"
    sample_df = pd.read_parquet(sample_path)
    sample_dates = pd.to_datetime(sample_df["timestamp_utc"])
    day_frames = [
        day_df[columns]
        for day_index, (_, day_df) in enumerate(sample_df.groupby(sample_dates.dt.date, sort=False))
        if day_index % date_steps == 0
    ]

    if len(day_frames) < n_days:
        raise ValueError(
            f"not enough sampled days in {sample_path.name}: requested {n_days}, "
            f"available {len(day_frames)} with date_steps={date_steps}"
        )

    return pd.concat(day_frames[:n_days], ignore_index=True)


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


def hypam_sweep(model: str, grid: str, n_days: int = 120, date_steps: int = 3):
    _, test_df, feature_columns, y_col = get_train_test_frames(
        target="base_load", model_family=model
    )
    train_df = load_feature_sample_df(
        start_id=100707,
        n_days=n_days,
        model_family=model,
        date_steps=date_steps,
    )

    X_train = train_df[feature_columns].to_numpy()
    y_train = train_df[y_col].to_numpy()
    X_test = test_df[feature_columns].to_numpy()
    y_test = test_df[y_col].to_numpy()

    out_dir = OUTPUT_DIR / "prediction" / "base_load" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{grid}_{100707}.csv"
    json_path = out_dir / f"{grid}.json"

    param_configs = build_param_grid(GRID_MAP[model][grid])
    rows = []
    print(
        f"Running CH hyperparameter sweep for model: {model}, grid: {grid}, "
        f"train_days: {n_days}, date_steps: {date_steps}"
    )
    for index, params in enumerate(param_configs, 1):
        estimator = build_model(model, "base_load", params)
        estimator.fit(X_train, y_train)
        rmse = float(root_mean_squared_error(y_test, estimator.predict(X_test)))
        print(f"  [{index:3d}/{len(param_configs)}] params={params} -> rmse={rmse:.5f}")
        rows.append({"params": json.dumps(params, sort_keys=True), "score": round(rmse, 5)})

    pd.DataFrame(rows).to_csv(out_path, index=False)
    with open(json_path, "w") as file:
        json.dump(GRID_MAP[model][grid], file, indent=2)
    print(f"Saved sweep results to {out_path}")
    print(f"Saved hypam grid to {json_path}")

if __name__ == "__main__":
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
    parser.add_argument(
        "--n_days",
        type=int,
        default=120,
        help="number of sampled CH training days",
    )
    parser.add_argument(
        "--date_steps",
        type=int,
        default=3,
        help="use every Nth calendar day from the diagonal sample",
    )
    args = parser.parse_args()

    models = [normalize_model(m) for m in args.model]
    grids = args.grid

    for model in models:
        for grid in grids:
            if grid not in GRID_MAP.get(model, {}):
                print(f"Skipping {model}/{grid}: grid not defined for model")
                continue
            hypam_sweep(
                n_days=args.n_days,
                date_steps=args.date_steps,
                model=model,
                grid=grid,
            )

