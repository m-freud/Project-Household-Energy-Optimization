'''
Combine the filtered csv files into a single table with the same format as the load table from the original dataset:

columns
timestamp_utc       | period | <player ids> -->
2022-12-31 00:00:00 | 1      | <load>
2022-12-31 00:15:00 | 2      | 0.51
2022-12-31 00:30:00 | 3      | 0.21
...

period runs 1-96 per day (00:00 -> 1, 23:45 -> 96).

Each player id column holds the "kWh_to_installation" values from the
corresponding <player_id>.csv file. Period 1 corresponds to 00:00, period 96
to 23:45 (15 minute resolution, 96 periods per day).
'''

import sqlite3
from pathlib import Path

import pandas as pd

smart_meters_dir = Path(__file__).parent / "smart_meter_data_filtered"
sqlite_path = Path(__file__).parents[2] / "sqlite" / "ch_smart_meters.db"

player_frames = []

for file in smart_meters_dir.glob("*.csv"):
    player_id = file.stem

    df = pd.read_csv(file, sep=";", parse_dates=["timestamp_utc"])
    timestamp = df["timestamp_utc"].dt.tz_localize(None)

    date_utc = timestamp.dt.date
    period = (timestamp.dt.hour * 4 + timestamp.dt.minute // 15) + 1

    player_df = pd.DataFrame(
        {
            "timestamp_utc": timestamp,
            "period": period,
            player_id: df["kWh_to_installation"],
        }
    )
    player_frames.append(player_df)
    print(f"{file.name}: {timestamp.min()} - {timestamp.max()}, {len(df)} rows")

combined = player_frames[0]
for player_df in player_frames[1:]:
    combined = combined.merge(player_df, on=["timestamp_utc", "period"], how="outer")

combined = combined.sort_values(["timestamp_utc", "period"]).reset_index(drop=True)
combined["timestamp_utc"] = combined["timestamp_utc"].astype(str)

with sqlite3.connect(sqlite_path) as conn:
    combined.to_sql("load", conn, if_exists="replace", index=False)

print(f"Wrote {len(combined)} rows, {len(combined.columns) - 2} players to {sqlite_path}")