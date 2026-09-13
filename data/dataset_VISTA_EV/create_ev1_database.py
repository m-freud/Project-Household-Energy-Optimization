import sqlite3
from pathlib import Path
import pandas as pd


# Paths
vista_dir = Path(__file__).parent
work_commute_csv = vista_dir / "work_commute_pairs.csv"

staging_dir = Path(__file__).parents[2] / "sqlite" / "staging"
staging_dir.mkdir(parents=True, exist_ok=True)

sqlite_path = staging_dir / "VISTA_ev1_status.db"


# Load work commute pairs
df = pd.read_csv(work_commute_csv)

# Ensure sorting by PERSID and TRIPNO
df = df.sort_values(["PERSID", "TRIPNO"]).reset_index(drop=True)


# Status table:
# rows = periods 1..96
# columns = PERSID
periods = list(range(1, 97))
ev_status_dict = {"period": periods}

# Distance table:
# persid | trip_no | distance
distance_rows = []


# Group by PERSID
for persid, group in df.groupby("PERSID"):
    trip1 = group[group["TRIPNO"] == 1]
    trip2 = group[group["TRIPNO"] == 2]

    # Skip incomplete commute pairs
    if trip1.empty or trip2.empty:
        continue

    persid_str = str(persid)

    # Trip 1: home -> work
    t1_startime = float(trip1["STARTIME"].iloc[0])
    t1_triptime = float(trip1["TRIPTIME"].iloc[0])
    t1_cumdist = float(trip1["CUMDIST"].iloc[0])

    # Trip 2: work -> home
    t2_startime = float(trip2["STARTIME"].iloc[0])
    t2_triptime = float(trip2["TRIPTIME"].iloc[0])
    t2_cumdist = float(trip2["CUMDIST"].iloc[0])

    # Store distances in long format
    distance_rows.append({
        "persid": persid_str,
        "trip_no": 1,
        "distance": t1_cumdist,
    })

    distance_rows.append({
        "persid": persid_str,
        "trip_no": 2,
        "distance": t2_cumdist,
    })

    # Convert times to 1-based period index (1..96)
    # 0 min -> period 1
    # 15 min -> period 2
    # ...
    start_period_1 = int(round(t1_startime / 15.0)) + 1
    trip_periods_1 = max(1, int(round(t1_triptime / 15.0)))

    start_period_2 = int(round(t2_startime / 15.0)) + 1
    trip_periods_2 = max(1, int(round(t2_triptime / 15.0)))

    # Status:
    # 0 = home
    # 1 = driving
    # 2 = parked_away / at work
    status_vec = [0] * 96

    # Trip 1: commute to work
    t1_start_idx = max(0, min(95, start_period_1 - 1))
    t1_end_idx = min(96, t1_start_idx + trip_periods_1)

    for p in range(t1_start_idx, t1_end_idx):
        status_vec[p] = 1

    # At work
    t2_start_idx = max(0, min(95, start_period_2 - 1))

    for p in range(t1_end_idx, t2_start_idx):
        status_vec[p] = 2

    # Trip 2: commute home
    t2_end_idx = min(96, t2_start_idx + trip_periods_2)

    for p in range(t2_start_idx, t2_end_idx):
        status_vec[p] = 1

    # Rest of day stays 0 = home
    ev_status_dict[persid_str] = status_vec


# Build DataFrames
result_status_df = pd.DataFrame(ev_status_dict)

result_distances_df = pd.DataFrame(
    distance_rows,
    columns=["persid", "trip_no", "distance"],
)


# Save to SQLite
with sqlite3.connect(sqlite_path) as conn:
    result_status_df.to_sql(
        "ev1_status",
        conn,
        if_exists="replace",
        index=False,
    )

    result_distances_df.to_sql(
        "ev1_distances",
        conn,
        if_exists="replace",
        index=False,
    )

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ev1_distances_person_trip
        ON ev1_distances (persid, trip_no)
    """)


print(f"Successfully processed {len(result_status_df.columns) - 1} persons.")
print(f"Status shape: {result_status_df.shape}")
print(f"Distance shape: {result_distances_df.shape}")
print(f"Output table 'ev1_status' written to {sqlite_path}")
print(f"Output table 'ev1_distances' written to {sqlite_path}")