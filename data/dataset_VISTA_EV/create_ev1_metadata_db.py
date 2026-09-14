"""
Calculate metadata for every EV1 commute profile in VISTA_ev1_status.db.

Input:
    VISTA_ev1_status.db

    tables:
        ev1_status
            period | <persid> | <persid> | ...

        ev1_distances
            persid | trip_no | distance

Output:
    VISTA_ev1_metadata.db

    table:
        ev1_metadata

Columns:
    persid

    trip_count
    total_distance
    mean_trip_distance
    max_trip_distance

    outbound_distance
    return_distance

    outbound_start
    outbound_end
    return_start
    return_end

    first_departure
    last_arrival

    driving_periods
    work_periods
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

ev1_db_path = staging_dir / "VISTA_ev1_status.db"
metadata_db_path = staging_dir / "VISTA_ev1_metadata.db"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with sqlite3.connect(ev1_db_path) as conn:
    status_df = pd.read_sql(
        "SELECT * FROM ev1_status",
        conn,
    )

    distance_df = pd.read_sql(
        "SELECT * FROM ev1_distances",
        conn,
    )


status_df = status_df.sort_values("period").reset_index(drop=True)

persids = [
    col
    for col in status_df.columns
    if col != "period"
]


print(f"Loaded {len(status_df)} status periods")
print(f"Found {len(persids)} EV1 persons")


# ---------------------------------------------------------------------------
# Calculate metadata
# ---------------------------------------------------------------------------

metadata_rows = []


for persid in persids:

    status = status_df[persid].astype(int)

    # -----------------------------------------------------------------------
    # Distances
    # -----------------------------------------------------------------------

    person_distances = distance_df[
        distance_df["persid"].astype(str) == str(persid)
    ].sort_values("trip_no")

    trip_count = len(person_distances)

    total_distance = person_distances["distance"].sum()
    mean_trip_distance = person_distances["distance"].mean()
    max_trip_distance = person_distances["distance"].max()

    outbound_rows = person_distances[
        person_distances["trip_no"] == 1
    ]

    return_rows = person_distances[
        person_distances["trip_no"] == 2
    ]

    outbound_distance = (
        float(outbound_rows["distance"].iloc[0])
        if not outbound_rows.empty
        else None
    )

    return_distance = (
        float(return_rows["distance"].iloc[0])
        if not return_rows.empty
        else None
    )


    # -----------------------------------------------------------------------
    # Status periods
    # -----------------------------------------------------------------------

    driving_mask = status == 1
    work_mask = status == 2
    home_mask = status == 0

    driving_periods = int(driving_mask.sum())
    work_periods = int(work_mask.sum())

    # Away = driving + parked at work
    away_periods = int((status != 0).sum())

    home_periods = int(home_mask.sum())

    away_fraction = away_periods / 96


    # -----------------------------------------------------------------------
    # Find driving windows
    # -----------------------------------------------------------------------

    driving_period_numbers = status_df.loc[
        driving_mask,
        "period"
    ].tolist()

    # Split driving periods into consecutive blocks.
    #
    # EV1 should normally result in exactly two blocks:
    #   block 1 = outbound commute
    #   block 2 = return commute

    driving_blocks = []

    for period in driving_period_numbers:

        if (
            not driving_blocks
            or period != driving_blocks[-1][-1] + 1
        ):
            driving_blocks.append([period])

        else:
            driving_blocks[-1].append(period)


    # -----------------------------------------------------------------------
    # Outbound commute
    # -----------------------------------------------------------------------

    if len(driving_blocks) >= 1:

        outbound_start = int(driving_blocks[0][0])
        outbound_end = int(driving_blocks[0][-1])

    else:

        outbound_start = None
        outbound_end = None


    # -----------------------------------------------------------------------
    # Return commute
    # -----------------------------------------------------------------------

    if len(driving_blocks) >= 2:

        return_start = int(driving_blocks[1][0])
        return_end = int(driving_blocks[1][-1])

    else:

        return_start = None
        return_end = None


    # -----------------------------------------------------------------------
    # General timing
    # -----------------------------------------------------------------------

    first_departure = outbound_start
    last_arrival = return_end


    # -----------------------------------------------------------------------
    # Store
    # -----------------------------------------------------------------------

    metadata_rows.append(
        {
            "persid": str(persid),

            "trip_count": trip_count,

            "total_distance": total_distance,
            "mean_trip_distance": mean_trip_distance,
            "max_trip_distance": max_trip_distance,

            "outbound_distance": outbound_distance,
            "return_distance": return_distance,

            "outbound_start": outbound_start,
            "outbound_end": outbound_end,

            "return_start": return_start,
            "return_end": return_end,

            "first_departure": first_departure,
            "last_arrival": last_arrival,

            "driving_periods": driving_periods,
            "work_periods": work_periods,
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
        "ev1_metadata",
        conn,
        if_exists="replace",
        index=False,
    )

    # One metadata row per person
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_ev1_metadata_persid
        ON ev1_metadata (persid)
    """)

    # Useful for mobility analysis / filtering
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ev1_metadata_total_distance
        ON ev1_metadata (total_distance)
    """)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print(f"Calculated metadata for {len(metadata_df)} EV1 persons")
print(f"Output: {metadata_db_path}")
print()
print(metadata_df.head())