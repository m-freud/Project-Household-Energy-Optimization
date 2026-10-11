"""
Select the candidate pool of CH load profiles for sampling.

A player-day from CH_loads.db.base_load_metadata is a candidate if its
total_energy and max_load lie within the [min, max] range of the original
profiles in energy.db.base_load_metadata.

Output:
    CH_loads.db
        table: base_load_pool   (subset of base_load_metadata, same columns)
"""

import sqlite3
from pathlib import Path


sqlite_dir = Path(__file__).parents[2] / "sqlite"

load_db_path = sqlite_dir / "staging" / "CH_loads.db"
energy_db_path = sqlite_dir / "energy.db"

SOURCE_TABLE = "base_load_metadata"
POOL_TABLE = "base_load_pool"


with sqlite3.connect(load_db_path) as conn:
    conn.execute("ATTACH DATABASE ? AS orig", (str(energy_db_path),))

    min_energy, max_energy, min_load, max_load = conn.execute(
        f"""
        SELECT MIN(total_energy), MAX(total_energy), MIN(max_load), MAX(max_load)
        FROM orig.{SOURCE_TABLE}
        """
    ).fetchone()

    conn.execute(f"DROP TABLE IF EXISTS {POOL_TABLE}")
    conn.execute(
        f"""
        CREATE TABLE {POOL_TABLE} AS
        SELECT *
        FROM main.{SOURCE_TABLE}
        WHERE total_energy BETWEEN ? AND ?
          AND max_load BETWEEN ? AND ?
        """,
        (min_energy, max_energy, min_load, max_load),
    )
    conn.execute(
        f"CREATE UNIQUE INDEX idx_{POOL_TABLE}_player_date ON {POOL_TABLE} (player_id, date)"
    )

    pool_size, total_size = conn.execute(
        f"SELECT (SELECT COUNT(*) FROM {POOL_TABLE}), (SELECT COUNT(*) FROM main.{SOURCE_TABLE})"
    ).fetchone()


print(f"total_energy range: {min_energy:.3f} - {max_energy:.3f} kWh")
print(f"max_load range:     {min_load:.3f} - {max_load:.3f} kW")
print(f"Pool: {pool_size} of {total_size} player-days in {POOL_TABLE} ({load_db_path})")
