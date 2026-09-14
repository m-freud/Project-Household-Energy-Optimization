"""
Calculate metadata for every PV system-day in DE_pv_gen.db.

Input:
    DE_pv_gen.db
        table: pv_gen

Output:
    DE_pv_gen_metadata.db
        table: pv_metadata

Columns:
    player_id
    date
    total_energy
    max_generation
    std_generation
    generation_factor
    peak_period
    generation_start
    generation_end
    active_periods
    ramp_std
    max_ramp
"""

import sqlite3
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

staging_dir = Path(__file__).parents[2] / "sqlite" / "staging"

pv_db_path = staging_dir / "DE_pv_gen.db"
metadata_db_path = staging_dir / "DE_pv_gen_metadata.db"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PERIOD_HOURS = 0.25

# Generation below this threshold is treated as zero.
# Avoids tiny meter/noise values affecting generation start/end.
ACTIVE_THRESHOLD_KW = 0.05


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with sqlite3.connect(pv_db_path) as conn:
    pv_df = pd.read_sql("SELECT * FROM pv_gen", conn)

pv_df["timestamp_utc"] = pd.to_datetime(pv_df["timestamp_utc"])
pv_df["date"] = pv_df["timestamp_utc"].dt.date

player_ids = [
    col
    for col in pv_df.columns
    if col not in {"timestamp_utc", "period", "date"}
]

print(f"Loaded {len(pv_df)} rows")
print(f"Found {len(player_ids)} PV systems")


# ---------------------------------------------------------------------------
# Calculate metadata
# ---------------------------------------------------------------------------

metadata_rows = []

for date, day_df in pv_df.groupby("date"):

    day_df = day_df.sort_values("period")

    for player_id in player_ids:

        values = day_df[player_id].dropna()

        # Only use complete 96-period days
        if len(values) != 96:
            continue

        values = values.astype(float)

        # ---------------------------------------------------------------
        # Basic generation statistics
        # ---------------------------------------------------------------

        total_energy = values.sum() * PERIOD_HOURS
        max_generation = values.max()
        mean_generation = values.mean()
        std_generation = values.std()

        # Skip days without meaningful PV generation
        if max_generation <= ACTIVE_THRESHOLD_KW:
            continue

        # Mean generation relative to daily peak.
        #
        # High value -> broad / relatively flat generation profile
        # Low value  -> narrow / peaky generation profile
        generation_factor = mean_generation / max_generation

        # ---------------------------------------------------------------
        # Timing
        # ---------------------------------------------------------------

        # Period with maximum generation
        peak_idx = values.idxmax()
        peak_period = int(day_df.loc[peak_idx, "period"])

        # Periods with meaningful PV generation
        active_mask = values > ACTIVE_THRESHOLD_KW
        active_indices = values.index[active_mask]

        # First and last period with meaningful generation
        generation_start = int(
            day_df.loc[active_indices[0], "period"]
        )

        generation_end = int(
            day_df.loc[active_indices[-1], "period"]
        )

        # Number of 15-minute periods with meaningful generation
        active_periods = int(active_mask.sum())

        # ---------------------------------------------------------------
        # Ramping / variability
        # ---------------------------------------------------------------

        ramps = values.diff().dropna()

        # Standard deviation of period-to-period changes
        ramp_std = ramps.std()

        # Largest absolute 15-minute change
        max_ramp = ramps.abs().max()

        # ---------------------------------------------------------------
        # Store
        # ---------------------------------------------------------------

        metadata_rows.append(
            {
                "player_id": str(player_id),
                "date": str(date),
                "total_energy": total_energy,
                "max_generation": max_generation,
                "std_generation": std_generation,
                "generation_factor": generation_factor,
                "peak_period": peak_period,
                "generation_start": generation_start,
                "generation_end": generation_end,
                "active_periods": active_periods,
                "ramp_std": ramp_std,
                "max_ramp": max_ramp,
            }
        )


metadata_df = pd.DataFrame(metadata_rows)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

with sqlite3.connect(metadata_db_path) as conn:

    metadata_df.to_sql(
        "pv_metadata",
        conn,
        if_exists="replace",
        index=False,
    )

    # Each PV system can only have one entry per date
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_pv_metadata_player_date
        ON pv_metadata (player_id, date)
    """)

    # Useful for matching/selecting profiles by peak generation
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_pv_metadata_max_generation
        ON pv_metadata (max_generation)
    """)

    # Useful for matching/selecting profiles by daily energy
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_pv_metadata_total_energy
        ON pv_metadata (total_energy)
    """)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print(f"Calculated metadata for {len(metadata_df)} PV player-days")
print(f"Output: {metadata_db_path}")
print()
print(metadata_df.head())