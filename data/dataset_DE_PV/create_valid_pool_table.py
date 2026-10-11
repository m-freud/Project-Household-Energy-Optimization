"""
Select the candidate pool of DE PV profiles for sampling.

A player-day from DE_pv_gen.db.pv_gen_metadata is a candidate if
active_periods is within [MIN_ACTIVE, MAX_ACTIVE]. total_energy and
max_generation are not filtered; they are normalized later.

Each candidate gets shift_periods, the shift that centers the profile on
CENTER_PERIOD (first and last non-zero period equally far from it).
All other columns and the profile data are left unchanged.

Output:
    DE_pv_gen.db
        table: pv_gen_pool   (pv_gen_metadata columns + shift_periods)
"""

import sqlite3
from pathlib import Path

import pandas as pd


pv_db_path = Path(__file__).parents[2] / "sqlite" / "staging" / "DE_pv_gen.db"

SOURCE_TABLE = "pv_gen_metadata"
POOL_TABLE = "pv_gen_pool"

MIN_ACTIVE = 46
MAX_ACTIVE = 54
CENTER_PERIOD = 62
PERIODS_PER_DAY = 96


with sqlite3.connect(pv_db_path) as conn:
    candidates = pd.read_sql(
        f"SELECT * FROM {SOURCE_TABLE} WHERE active_periods BETWEEN ? AND ?",
        conn,
        params=(MIN_ACTIVE, MAX_ACTIVE),
    )
    total_size = conn.execute(f"SELECT COUNT(*) FROM {SOURCE_TABLE}").fetchone()[0]

    # Odd start+end has no exact center; floor division rounds the midpoint down.
    candidates["shift_periods"] = (
        CENTER_PERIOD - (candidates["generation_start"] + candidates["generation_end"]) // 2
    )

    # Drop days that would be shifted out of the 1..96 day
    shifted_start = candidates["generation_start"] + candidates["shift_periods"]
    shifted_end = candidates["generation_end"] + candidates["shift_periods"]
    pool = candidates[(shifted_start >= 1) & (shifted_end <= PERIODS_PER_DAY)]

    pool.to_sql(POOL_TABLE, conn, if_exists="replace", index=False)
    conn.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{POOL_TABLE}_player_date ON {POOL_TABLE} (player_id, date)"
    )


print(f"active_periods range: {MIN_ACTIVE} - {MAX_ACTIVE}")
print(f"Pool: {len(pool)} of {total_size} player-days in {POOL_TABLE} ({pv_db_path})")
print(pool["shift_periods"].describe())
