from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from training.features._regression import (  # noqa: E402
    _add_accel_feature,
    _add_delta_features,
    _add_history_average_features,
    _add_std_features,
    _round_float_features,
)
from training.features._shared import (  # noqa: E402
    _add_lag_features,
    _add_next_value_target,
    _add_trig_time_features,
)


DEFAULT_DB_PATH = Path(__file__).parents[2] / "sqlite" / "ch_smart_meters.db"
DEFAULT_TABLE_NAME = "load_features"


def _get_household_ids(db_path: Path) -> list[int]:
    with sqlite3.connect(db_path) as conn:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(load)").fetchall()]
    return [int(col) for col in cols if col not in {"timestamp_utc", "period"}]


def _contains_nan(df: pd.DataFrame) -> bool:
    return bool(df.isna().any().any())


def _build_feature_day_frame(day_df: pd.DataFrame) -> pd.DataFrame:
    player_day_df = pd.DataFrame(
        {
            "timestamp_utc": day_df["timestamp_utc"].to_numpy(),
            "date": day_df["date"].to_numpy(),
            "period": day_df["period"].to_numpy(),
            "timestep": day_df["period"].to_numpy(),
            "household_id": int(day_df["household_id"].iloc[0]),
            "base_load": day_df["base_load"].to_numpy(),
        }
    )

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
    player_day_df = _round_float_features(player_day_df, digits=3)
    return player_day_df


def _build_feature_table_for_household(load_df: pd.DataFrame, household_id: int) -> pd.DataFrame:
    household_df = load_df[["timestamp_utc", "period", str(household_id)]].copy()
    household_df = household_df.rename(columns={str(household_id): "base_load"})
    household_df["household_id"] = int(household_id)
    household_df["date"] = pd.to_datetime(household_df["timestamp_utc"]).dt.date.astype(str)

    frames: list[pd.DataFrame] = []
    for _, day_df in household_df.groupby("date", sort=False):
        if len(day_df) != 96 or _contains_nan(day_df[["base_load"]]):
            continue

        frames.append(_build_feature_day_frame(day_df))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def create_feature_table(
    db_path: Path = DEFAULT_DB_PATH,
    table_name: str = DEFAULT_TABLE_NAME,
    household_ids: list[int] | None = None,
    overwrite: bool = True,
    max_households: int | None = None,
) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        load_df = pd.read_sql("SELECT * FROM load", conn)

    if household_ids is None:
        household_ids = _get_household_ids(db_path)
    if max_households is not None:
        household_ids = household_ids[:max_households]

    column_order = [
        "timestamp_utc",
        "date",
        "period",
        "household_id",
        "timestep",
        "base_load",
        "time_sin",
        "time_cos",
        "base_load_lag_1",
        "base_load_lag_2",
        "base_load_lag_4",
        "base_load_lag_8",
        "base_load_lag_12",
        "base_load_ma_2",
        "base_load_ma_4",
        "base_load_ma_8",
        "base_load_ma_16",
        "base_load_std_4",
        "base_load_std_8",
        "base_load_delta_1",
        "base_load_delta_2",
        "base_load_accel",
        "next_value",
    ]

    with sqlite3.connect(db_path) as conn:
        if overwrite:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.commit()

        empty_df = pd.DataFrame(columns=column_order)
        if not conn.execute(f"SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone():
            empty_df.to_sql(table_name, conn, index=False, if_exists="replace")
            conn.commit()

        total = len(household_ids)
        rows_written = 0
        for idx, household_id in enumerate(household_ids, start=1):
            print(f"[household {idx}/{total}] starting household_id={household_id}")
            household_feature_df = _build_feature_table_for_household(load_df, household_id)
            if household_feature_df.empty:
                print(f"[household {idx}/{total}] household_id={household_id}: no valid rows, skipping")
                continue

            household_feature_df = household_feature_df[[col for col in column_order if col in household_feature_df.columns]].copy()
            household_feature_df = household_feature_df.sort_values(["household_id", "timestamp_utc"]).reset_index(drop=True)
            household_feature_df.to_sql(table_name, conn, index=False, if_exists="append")
            rows_written += len(household_feature_df)
            print(f"[household {idx}/{total}] household_id={household_id}: wrote {len(household_feature_df)} rows")

        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_household_date_period "
            f"ON {table_name}(household_id, date, period)"
        )
        conn.commit()

    if rows_written == 0:
        raise ValueError("No valid feature rows were generated.")

    final_df = pd.read_sql(f"SELECT * FROM {table_name}", sqlite3.connect(db_path))
    return final_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute base-load feature rows for CH smart-meter data.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite database that contains the raw load table.")
    parser.add_argument("--table", type=str, default=DEFAULT_TABLE_NAME, help="Table name for the feature output.")
    parser.add_argument("--household-ids", nargs="*", type=int, default=None, help="Optional subset of household IDs to process.")
    parser.add_argument("--max-households", type=int, default=None, help="Optional cap on processed households for a quick test run.")
    parser.add_argument("--overwrite", action="store_true", help="Drop and recreate the target table before writing.")
    args = parser.parse_args()

    if not args.db.exists():
        raise FileNotFoundError(f"Database not found: {args.db}")

    table_name = args.table
    if not table_name.replace("_", "").isalnum():
        raise ValueError(f"Invalid table name: {table_name!r}")

    feature_df = create_feature_table(
        db_path=args.db,
        table_name=table_name,
        household_ids=args.household_ids,
        overwrite=args.overwrite,
        max_households=args.max_households,
    )

    print(f"Wrote {len(feature_df)} feature rows to {args.db}::{table_name}")
    print(f"Households: {feature_df['household_id'].nunique()}")
    print(f"Dates: {feature_df['date'].nunique()}")


if __name__ == "__main__":
    main()
