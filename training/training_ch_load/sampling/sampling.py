import sqlite3
from pathlib import Path
import sys

import pandas as pd
import numpy as np

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from training.features._regression import (  # noqa: E402
    _add_history_average_features,
    _add_std_features,
    _add_delta_features,
    _add_accel_feature,
    _round_float_features,
)
from training.features._shared import (  # noqa: E402
    _add_trig_time_features,
    _add_lag_features,
    _add_next_value_target,
)

SQLITE_PATH = Path(__file__).parents[3] / "sqlite" / "ch_smart_meters.db"
DIAG_FEATURE_DIR = Path(__file__).parent / "diag_feature_samples"
RANDOM_FEATURE_DIR = Path(__file__).parent / "random_feature_samples"


ALL_IDS = ['100354', '100707', '102016', '103995', '104481', '105366', '106298', '106793', '107448', '108517', '108795', '109970', '110260', '111381','112377', '113377', '115937', '116320', '118984', '119559', '120517', '120691', '121500', '121555', '122670', '123129', '125463', '127225', '127524', '128843', '129071', '130253', '131514', '132256', '132967', '133165', '133996', '136407', '137077', '138414', '139410', '140211', '140649', '142975', '142999', '144492', '146624', '147049', '147147', '151683', '151740', '152601', '155209', '157904', '159093', '160216', '160431', '161573', '162680', '163329', '163595', '163677', '163817', '164897', '165786', '166084', '166206', '167494', '168165', '169310', '170013', '171530', '172262', '172510', '174968', '176249', '176350', '179723', '180146', '180931', '183144', '184973', '185196', '187422', '188093', '188131', '188208', '189416', '191628', '191701', '192815', '198650', '199888']

def get_id_list():
    # get column names of load table of ch_smart_meters.db
    with sqlite3.connect(SQLITE_PATH) as conn:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(load)").fetchall()]
    return [c for c in cols if c not in ("timestamp_utc", "period")]


def __get_diagonal_sample(start_idx, n_days) -> pd.DataFrame:
    """
    DEPRECATED
    Get a diagonal sample of load data for a specified number of days starting from a given index.

    Args:
        start_idx (int): The starting index for selecting households.
        n_days (int): The number of days to sample.

    Returns:
        pd.DataFrame: A DataFrame containing the sampled load data with columns ["timestamp_utc", "period", "household_id", "load"].
    """
    with sqlite3.connect(SQLITE_PATH) as conn:
        load_df = pd.read_sql("SELECT * FROM load", conn)
    load_df = load_df[
        load_df["timestamp_utc"].between(
            "2023-01-01 00:00:00",
            "2024-12-30 23:45:00" # 2 years
        )
    ]

    player_cols = load_df.columns[2:]
    samples = []

    for day in range(n_days):
        player = player_cols[(start_idx + day) % len(player_cols)]

        start = day * 96
        end = start + 96

        day_df = load_df.iloc[start:end][
            ["timestamp_utc", "period", player]
        ].copy()

        day_df = day_df.rename(columns={player: "load"})
        day_df.insert(2, "household_id", int(player))

        samples.append(day_df)

    return pd.concat(samples, ignore_index=True)


def contains_nan(df: pd.DataFrame):
    return bool(df.isna().any().any())


def get_random_sample(seed: int, n_days: int) -> pd.DataFrame:
    with sqlite3.connect(SQLITE_PATH) as conn:
        load_df = pd.read_sql("SELECT * FROM load", conn)

    load_df = load_df[
        load_df["timestamp_utc"].between(
            "2023-01-01 00:00:00",
            "2024-12-30 23:45:00"
        )
    ].copy()

    load_df["date"] = pd.to_datetime(load_df["timestamp_utc"]).dt.date

    player_cols = load_df.columns[2:-1]
    dates = load_df["date"].unique()

    # deterministic random generator
    rng = np.random.default_rng(seed=seed)

    # all possible (player, day) combinations
    combinations = [
        (player, date)
        for player in player_cols
        for date in dates
    ]

    # deterministic shuffle
    rng.shuffle(combinations)

    samples = []

    for player, date in combinations:
        day_df = load_df.loc[
            load_df["date"] == date,
            ["timestamp_utc", "period", player]
        ].copy()

        # skip incomplete / broken days
        if len(day_df) != 96 or contains_nan(day_df):
            continue

        day_df = day_df.rename(columns={player: "load"})
        day_df.insert(2, "household_id", int(player))

        samples.append(day_df)

        if len(samples) == n_days:
            break

    if len(samples) != n_days:
        raise ValueError(f"Requested {n_days} valid player-days, found {len(samples)}")

    return pd.concat(samples, ignore_index=True)


def create_sample_feature_df(
    sample_df: pd.DataFrame,
    round_values: bool = True,
    output_dir: Path = DIAG_FEATURE_DIR,
    tag: str = "abc123",
) -> pd.DataFrame:

    day_frames = []
    for _, day_df in sample_df.groupby(
        pd.to_datetime(sample_df["timestamp_utc"]).dt.date
    ):
        if contains_nan(day_df):
            print(f"Skipping day with NaN values: {day_df['timestamp_utc'].iloc[0]}, player {day_df['household_id'].iloc[0]}")
            continue
        
        player_day_df = pd.DataFrame({
            "timestamp_utc": day_df["timestamp_utc"].to_numpy(),
            "timestep": day_df["period"].to_numpy(),
            "household_id": day_df["household_id"].to_numpy(),
            "base_load": day_df["load"].to_numpy(),
        })

        player_day_df = _add_trig_time_features(player_day_df)
        player_day_df = _add_lag_features(
            player_day_df,
            source_column="base_load",
            group_cols=("household_id",),
            lags=(1, 2, 4, 8, 12),
            pad_value=-1.0,
            add_pad_flags=False,
            output_prefix="base_load_lag",
            dtype=float,
        )
        player_day_df = _add_history_average_features(
            player_day_df,
            windows=(2, 4, 8, 16),
            value_column="base_load",
            prefix="base_load",
        )
        player_day_df = _add_std_features(
            player_day_df,
            windows=(4, 8),
            value_column="base_load",
            prefix="base_load",
        )
        player_day_df = _add_delta_features(
            player_day_df,
            value_column="base_load",
            prefix="base_load",
        )
        player_day_df = _add_accel_feature(
            player_day_df,
            prefix="base_load",
        )
        player_day_df = _add_next_value_target(
            player_day_df,
            source_column="base_load",
            group_cols=("household_id",),
            target_column="next_value",
            fill_value=0.0,
            dtype=float,
        )
        if round_values:
            player_day_df = _round_float_features(player_day_df, digits=3)

        day_frames.append(player_day_df)

    feature_df = pd.concat(day_frames, ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(output_dir / f"{tag}_random_features.parquet", index=False)

    return feature_df


if __name__ == '__main__':
    for seed in range(10):
        print(f"creating sample feature df for seed={seed}")
        random_sample_df = get_random_sample(seed, n_days=120)
        create_sample_feature_df(sample_df=random_sample_df, output_dir=RANDOM_FEATURE_DIR, tag=f"seed_{seed}")