
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
from training.training_ch_load.sampling.sampling import ALL_IDS, RANDOM_FEATURE_DIR  # noqa

OUTPUT_DIR = Path(__file__).parent

xgb_best_params = {"learning_rate": 0.03, "max_depth": 2, "n_estimators": 200} # best params from load sweep on original data

def load_feature_sample_df(seed=1, n_days=120, model_family="xgboost", split="train"):
    features = MODEL_FEATURES_BY_FAMILY[model_family]["base_load"]
    columns = features + ["next_value"]
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    sample_path = RANDOM_FEATURE_DIR / split / f"{split}_seed_{seed}_random_features.parquet"
    sample_df = pd.read_parquet(sample_path)
    sample_dates = pd.to_datetime(sample_df["timestamp_utc"])
    day_frames = [
        day_df[columns]
        for _, day_df in sample_df.groupby(
            [sample_df["household_id"], sample_dates.dt.date], sort=False
        )
    ]

    if len(day_frames) < n_days:
        raise ValueError(
            f"not enough sampled days in {sample_path.name}: requested {n_days}, "
            f"available {len(day_frames)}"
        )

    return pd.concat(day_frames[:n_days], ignore_index=True)


def sweep_n_days(
    model_family="xgboost",
    train_seed=1,
    test_seeds=(0, 1, 2, 3),
    n_test_days=120,
    max_n_days=120,
    plateau_tol=1e-3,
    save_plot=True,
):
    features = MODEL_FEATURES_BY_FAMILY[model_family]["base_load"]
    params = xgb_best_params if model_family == "xgboost" else {}
    if not test_seeds:
        raise ValueError("test_seeds must contain at least one test seed")
    if max_n_days < 1 or max_n_days > 120:
        raise ValueError("max_n_days must be between 1 and 120 for the available random samples")

    test_df = pd.concat(
        [
            load_feature_sample_df(
                seed=test_seed,
                n_days=n_test_days,
                model_family=model_family,
                split="test",
            )
            for test_seed in test_seeds
        ],
        ignore_index=True,
    )
    X_test, y_test = test_df[features], test_df["next_value"]

    rows = []
    prev_rmse = None
    for n_days in range(1, max_n_days + 1, 1):
        train_df = load_feature_sample_df(
            seed=train_seed,
            n_days=n_days,
            model_family=model_family,
            split="train",
        )

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
    output_stem = f"{model_family}_train_seed_{train_seed}_train_size_sweep"
    result_df.to_csv(output_dir / f"{output_stem}.csv", index=False)

    if save_plot:
        fig, ax = plt.subplots()
        ax.plot(result_df["n_train_days"], result_df["rmse"], marker="o")
        ax.set_xlabel("n_train_days")
        ax.set_ylabel("rmse")
        ax.set_title(f"{model_family} base_load random-sample train size sweep")
        fig.tight_layout()
        fig.savefig(output_dir / f"{output_stem}.png")
        plt.close(fig)

    return result_df


def sweep_n_days_for_seeds(
    model_family="xgboost",
    train_seeds=tuple(range(12)),
    test_seeds=(0, 1, 2, 3),
    n_test_days=120,
    max_n_days=120,
    plateau_tol=1e-3,
):
    result_frames = []
    for train_seed in train_seeds:
        print(f"Running train-size sweep for train_seed={train_seed}")
        result_df = sweep_n_days(
            model_family=model_family,
            train_seed=train_seed,
            test_seeds=test_seeds,
            n_test_days=n_test_days,
            max_n_days=max_n_days,
            plateau_tol=plateau_tol,
            save_plot=False,
        )
        result_frames.append(result_df.assign(train_seed=train_seed))

    combined_df = pd.concat(result_frames, ignore_index=True)
    output_dir = Path(__file__).parent / "train_size_sweeps"
    output_stem = f"{model_family}_multi_seed_train_size_sweep"
    combined_df.to_csv(output_dir / f"{output_stem}.csv", index=False)

    fig, ax = plt.subplots()
    for train_seed, seed_df in combined_df.groupby("train_seed", sort=True):
        ax.plot(
            seed_df["n_train_days"],
            seed_df["rmse"],
            marker="o",
            label=f"train seed {train_seed}",
        )
    ax.set_xlabel("n_train_days")
    ax.set_ylabel("rmse")
    ax.set_title(f"{model_family} base_load train size by random train seed")
    ax.legend(title="train sample")
    fig.tight_layout()
    fig.savefig(output_dir / f"{output_stem}.png")
    plt.close(fig)

    return combined_df


def hypam_sweep(model: str, grid: str, n_days: int = 120, seed: int = 1):
    _, test_df, feature_columns, y_col = get_train_test_frames(
        target="base_load", model_family=model
    )
    train_df = load_feature_sample_df(
        seed=seed,
        n_days=n_days,
        model_family=model,
        split="train",
    )

    X_train = train_df[feature_columns].to_numpy()
    y_train = train_df[y_col].to_numpy()
    X_test = test_df[feature_columns].to_numpy()
    y_test = test_df[y_col].to_numpy()

    out_dir = OUTPUT_DIR / "pred" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"{grid}_seed_{seed}"
    out_path = out_dir / f"{output_stem}.csv"
    json_path = out_dir / f"{output_stem}.json"

    param_configs = build_param_grid(GRID_MAP[model][grid])
    rows = []
    print(
        f"Running CH hyperparameter sweep for model: {model}, grid: {grid}, "
        f"train_seed: {seed}, train_days: {n_days}"
    )
    for index, params in enumerate(param_configs, 1):
        estimator = build_model(model, "base_load", params)
        estimator.fit(X_train, y_train)
        rmse = float(root_mean_squared_error(y_test, estimator.predict(X_test)))
        print(f"  [{index:3d}/{len(param_configs)}] params={params} -> rmse={rmse:.5f}")
        rows.append({"params": json.dumps(params, sort_keys=True), "score": round(rmse, 5)})

    pd.DataFrame(rows).to_csv(out_path, index=False)
    with open(json_path, "w") as file:
        json.dump(
            {
                "grid": GRID_MAP[model][grid],
                "target": "base_load",
                "model": model,
                "ch_train_split": "train",
                "ch_train_seed": seed,
                "ch_training_days": n_days,
            },
            file,
            indent=2,
        )
    print(f"Saved sweep results to {out_path}")
    print(f"Saved hypam grid to {json_path}")


if __name__ == "__main__":
    # lets do the n day saturation sweep first
    sweep_n_days_for_seeds(
    train_seeds=tuple(range(12)),
    test_seeds=(0, 1, 2, 3),
    max_n_days=120,
)


    exit()
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
        "--seed",
        type=int,
        default=1,
        help="random CH training sample seed",
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
                seed=args.seed,
                model=model,
                grid=grid,
            )

