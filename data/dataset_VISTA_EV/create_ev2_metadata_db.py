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
    VISTA_ev2_metadata.db

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

staging_dir = Path(__file__).parents[2] / "sqlite" / "staging"

ev2_db_path = staging_dir / "VISTA_ev2_status.db"
metadata_db_path = staging_dir / "VISTA_ev2_metadata.db"


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
# Calculate metadata
# ---------------------------------------------------------------------------

metadata_rows = []


for persid in persids:

    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------

    person_status = status_df[
        status_df["persid"] == persid
    ].sort_values("period")

    # Only use complete profiles
    if len(person_status) != 96:
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


metadata_df = pd.DataFrame(metadata_rows)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

with sqlite3.connect(metadata_db_path) as conn:

    metadata_df.to_sql(
        "ev2_metadata",
        conn,
        if_exists="replace",
        index=False,
    )

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

print()
print(f"Calculated metadata for {len(metadata_df)} EV2 persons")
print(f"Output: {metadata_db_path}")
print()
print(metadata_df.head())