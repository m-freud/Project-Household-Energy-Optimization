import sqlite3
from pathlib import Path
import pandas as pd

# Paths
vista_dir = Path(__file__).parent
casual_commute_csv = vista_dir / "casual_commutes.csv"

staging_dir = Path(__file__).parents[2] / "sqlite" / "staging"
staging_dir.mkdir(parents=True, exist_ok=True)

sqlite_path = staging_dir / "VISTA_ev2_status.db"

# Load and sort
df = pd.read_csv(casual_commute_csv)
df = df.sort_values(["PERSID", "TRIPNO"]).reset_index(drop=True)

status_rows = []
distance_rows = []

grouped = df.groupby("PERSID")

for persid, group in grouped:
    trips = group.sort_values("TRIPNO").reset_index(drop=True)

    if trips.empty:
        continue

    persid_str = str(persid)

    # 0 = home
    # 1 = driving
    # 2 = parked_away
    status_vec = [0] * 96

    for i, trip in trips.iterrows():
        trip_no = int(trip["TRIPNO"])

        # Store trip distance
        distance_rows.append({
            "persid": persid_str,
            "trip_no": trip_no,
            "distance": float(trip["CUMDIST"]),
        })

        # Trip timing
        startime = float(trip["STARTIME"])
        triptime = float(trip["TRIPTIME"])

        start_period = int(round(startime / 15.0)) + 1
        trip_periods = max(1, int(round(triptime / 15.0)))

        trip_start_idx = max(0, min(95, start_period - 1))
        trip_end_idx = min(96, trip_start_idx + trip_periods)

        # Driving
        for p in range(trip_start_idx, trip_end_idx):
            status_vec[p] = 1

        # State at destination
        dest_purp = str(trip.get("DESTPURP1", ""))
        stay_status = 0 if dest_purp == "Go Home" else 2

        # Stay until next trip
        if i + 1 < len(trips):
            next_startime = float(trips.iloc[i + 1]["STARTIME"])
            next_start_period = int(round(next_startime / 15.0)) + 1
            next_start_idx = max(
                trip_end_idx,
                min(96, next_start_period - 1),
            )
        else:
            next_start_idx = 96

        for p in range(trip_end_idx, next_start_idx):
            status_vec[p] = stay_status

    # Convert 96 periods to long format
    for period, status in enumerate(status_vec, start=1):
        status_rows.append({
            "persid": persid_str,
            "period": period,
            "status": status,
        })


result_status_df = pd.DataFrame(status_rows)
result_distances_df = pd.DataFrame(distance_rows)


# Save
with sqlite3.connect(sqlite_path) as conn:
    result_status_df.to_sql(
        "ev2_status",
        conn,
        if_exists="replace",
        index=False,
    )

    result_distances_df.to_sql(
        "ev2_distances",
        conn,
        if_exists="replace",
        index=False,
    )

    # Useful indexes
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ev2_status_person_period
        ON ev2_status (persid, period)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ev2_distances_person_trip
        ON ev2_distances (persid, trip_no)
    """)


print(f"Successfully processed {df['PERSID'].nunique()} persons.")
print(f"Status rows: {len(result_status_df)}")
print(f"Distance rows: {len(result_distances_df)}")
print(f"Written to {sqlite_path}")