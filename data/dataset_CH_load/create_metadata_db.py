"""
Calculate metadata for every player-day in CH_loads.db.

Input table:
    load

    timestamp_utc       | period | <player ids> -->
    2022-12-31 00:00:00 | 1      | 0.51
    ...
    2022-12-31 23:45:00 | 96     | 0.21

Output:
    CH_load_metadata.db
        table: load_metadata

Columns:
    player_id
    date
    total_energy
    max_load
    std_load
    load_factor
    peak_period
    morning_mean
    evening_mean
    ramp_std
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

staging_dir = Path(__file__).parents[2] / "sqlite" / "staging"

load_db_path = staging_dir / "CH_loads.db"
metadata_db_path = staging_dir / "CH_load_metadata.db"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# 06:00 - 11:45
MORNING_PERIODS = range(25, 49)

# 17:00 - 22:45
EVENING_PERIODS = range(69, 93)

PERIOD_HOURS = 0.25


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with sqlite3.connect(load_db_path) as conn:
    load_df = pd.read_sql("SELECT * FROM load", conn)

load_df["timestamp_utc"] = pd.to_datetime(load_df["timestamp_utc"])
load_df["date"] = load_df["timestamp_utc"].dt.date

player_ids = [
    col
    for col in load_df.columns
    if col not in {"timestamp_utc", "period", "date"}
]

print(f"Loaded {len(load_df)} rows")
print(f"Found {len(player_ids)} players")


# ---------------------------------------------------------------------------
# Calculate metadata
# ---------------------------------------------------------------------------

metadata_rows = []

for date, day_df in load_df.groupby("date"):

    # Ensure correct temporal order
    day_df = day_df.sort_values("period")

    for player_id in player_ids:

        values = day_df[player_id].dropna()

        # Skip incomplete player-days
        if len(values) != 96:
            continue

        values = values.astype(float)

        total_energy = values.sum() * PERIOD_HOURS
        max_load = values.max()
        mean_load = values.mean()
        std_load = values.std()

        # mean / peak:
        # close to 1 -> flat profile
        # low -> peaky profile
        load_factor = (
            mean_load / max_load
            if max_load > 0
            else 0.0
        )

        # Period at which maximum load occurs
        peak_idx = values.idxmax()
        peak_period = int(day_df.loc[peak_idx, "period"])

        # Time-of-day means
        morning_mean = day_df.loc[
            day_df["period"].isin(MORNING_PERIODS),
            player_id,
        ].mean()

        evening_mean = day_df.loc[
            day_df["period"].isin(EVENING_PERIODS),
            player_id,
        ].mean()

        # Volatility of period-to-period changes
        ramp_std = values.diff().dropna().std()

        metadata_rows.append(
            {
                "player_id": str(player_id),
                "date": str(date),
                "total_energy": total_energy,
                "max_load": max_load,
                "std_load": std_load,
                "load_factor": load_factor,
                "peak_period": peak_period,
                "morning_mean": morning_mean,
                "evening_mean": evening_mean,
                "ramp_std": ramp_std,
            }
        )


metadata_df = pd.DataFrame(metadata_rows)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

with sqlite3.connect(metadata_db_path) as conn:

    metadata_df.to_sql(
        "load_metadata",
        conn,
        if_exists="replace",
        index=False,
    )

    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_load_metadata_player_date
        ON load_metadata (player_id, date)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_load_metadata_max_load
        ON load_metadata (max_load)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_load_metadata_total_energy
        ON load_metadata (total_energy)
    """)


print()
print(f"Calculated metadata for {len(metadata_df)} player-days")
print(f"Output: {metadata_db_path}")
print()
print(metadata_df.head())