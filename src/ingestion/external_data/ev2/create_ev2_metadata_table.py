"""
Calculate metadata for every EV2 mobility profile in VISTA_ev2_status.db.

Input:
    VISTA_ev2_status.db

    tables:
        ev2_status
            persid | period | status

        ev2_distances
            persid | trip_no | distance

Output:
    VISTA_ev2_status.db (same DB as the input)

    table:
        ev2_metadata

Columns:
    persid
    trip_count
    total_distance
    mean_trip_distance
    max_trip_distance
    first_departure
    last_arrival
    driving_periods
    away_periods
    home_periods
    away_fraction
"""

import sqlite3
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
staging_dir = ROOT_DIR / "sqlite" / "staging"

ev2_db_path = staging_dir / "VISTA_ev2_status.db"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with sqlite3.connect(ev2_db_path) as conn:

    status_df = pd.read_sql(
        "SELECT * FROM ev2_status",
        conn,
    )

    distance_df = pd.read_sql(
        "SELECT * FROM ev2_distances",
        conn,
    )


status_df["persid"] = status_df["persid"].astype(str)
distance_df["persid"] = distance_df["persid"].astype(str)

status_df = status_df.sort_values(
    ["persid", "period"]
).reset_index(drop=True)


persids = status_df["persid"].unique()

print(f"Loaded {len(status_df)} status rows")
print(f"Found {len(persids)} EV2 persons")


# ---------------------------------------------------------------------------
# Prepare the output table for incremental writes
# ---------------------------------------------------------------------------

with sqlite3.connect(ev2_db_path) as conn:
    conn.execute("DROP TABLE IF EXISTS ev2_metadata")
    conn.execute("""
        CREATE TABLE ev2_metadata (
            persid TEXT PRIMARY KEY,
            trip_count INTEGER,
            total_distance REAL,
            mean_trip_distance REAL,
            max_trip_distance REAL,
            first_departure INTEGER,
            last_arrival INTEGER,
            driving_periods INTEGER,
            away_periods INTEGER,
            home_periods INTEGER,
            away_fraction REAL
        )
    """)


# ---------------------------------------------------------------------------
# Calculate metadata
# ---------------------------------------------------------------------------

batch_size = 500
metadata_rows = []
metadata_count = 0


def save_metadata_batch(rows: list[dict]) -> int:
    if not rows:
        return 0

    metadata_df = pd.DataFrame(rows)
    with sqlite3.connect(ev2_db_path) as conn:
        metadata_df.to_sql(
            "ev2_metadata",
            conn,
            if_exists="append",
            index=False,
        )
        conn.commit()
    return len(metadata_df)


for i, persid in enumerate(persids):
    if (i)**0.5 == int((i)**0.5):
        print(f"Processing {i+1}/{len(persids)}: persid={persid}")

    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------

    person_status = status_df[
        status_df["persid"] == persid
    ].sort_values("period")

    # Only use complete profiles
    if len(person_status) != 96:
        if (i + 1) % batch_size == 0 or i + 1 == len(persids):
            metadata_count += save_metadata_batch(metadata_rows)
            metadata_rows.clear()
            print(f"Saved metadata through {i + 1}/{len(persids)} persons")
        continue

    status = person_status["status"].astype(int)


    # -----------------------------------------------------------------------
    # Distances
    # -----------------------------------------------------------------------

    person_distances = distance_df[
        distance_df["persid"] == persid
    ].sort_values("trip_no")

    trip_count = len(person_distances)

    total_distance = person_distances["distance"].sum()
    mean_trip_distance = person_distances["distance"].mean()
    max_trip_distance = person_distances["distance"].max()


    # -----------------------------------------------------------------------
    # Status statistics
    # -----------------------------------------------------------------------

    driving_periods = int(
        (status == 1).sum()
    )

    # Away includes both driving and parked away
    away_periods = int(
        (status != 0).sum()
    )

    home_periods = int(
        (status == 0).sum()
    )

    away_fraction = away_periods / 96


    # -----------------------------------------------------------------------
    # First departure
    # -----------------------------------------------------------------------

    driving_rows = person_status[
        person_status["status"] == 1
    ]

    if not driving_rows.empty:

        first_departure = int(
            driving_rows["period"].iloc[0]
        )

        # Last driving period of the final trip
        last_arrival = int(
            driving_rows["period"].iloc[-1]
        )

    else:

        first_departure = None
        last_arrival = None


    # -----------------------------------------------------------------------
    # Store
    # -----------------------------------------------------------------------

    metadata_rows.append(
        {
            "persid": persid,

            "trip_count": trip_count,

            "total_distance": total_distance,
            "mean_trip_distance": mean_trip_distance,
            "max_trip_distance": max_trip_distance,

            "first_departure": first_departure,
            "last_arrival": last_arrival,

            "driving_periods": driving_periods,
            "away_periods": away_periods,
            "home_periods": home_periods,
            "away_fraction": away_fraction,
        }
    )

    if (i + 1) % batch_size == 0 or i + 1 == len(persids):
        metadata_count += save_metadata_batch(metadata_rows)
        metadata_rows.clear()
        print(f"Saved metadata through {i + 1}/{len(persids)} persons")


# ---------------------------------------------------------------------------
# Create indexes after all batches are committed
# ---------------------------------------------------------------------------

with sqlite3.connect(ev2_db_path) as conn:
    # One metadata row per person
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ev2_metadata_persid
        ON ev2_metadata (persid)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ev2_metadata_total_distance
        ON ev2_metadata (total_distance)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ev2_metadata_trip_count
        ON ev2_metadata (trip_count)
    """)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

with sqlite3.connect(ev2_db_path) as conn:
    metadata_preview = pd.read_sql(
        "SELECT * FROM ev2_metadata LIMIT 5",
        conn,
    )

print()
print(f"Calculated metadata for {metadata_count} EV2 persons")
print(f"Output: table ev2_metadata in {ev2_db_path}")
print()
print(metadata_preview)