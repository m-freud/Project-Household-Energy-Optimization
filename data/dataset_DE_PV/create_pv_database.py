import sqlite3
from pathlib import Path
import pandas as pd

pv_csv_path = Path(__file__).parent / "pv_data_non_empty.csv"
staging_dir = Path(__file__).parents[2] / "sqlite" / "staging"
staging_dir.mkdir(parents=True, exist_ok=True)
sqlite_path = staging_dir / "DE_pv_gen.db"

# 1. Read raw data
df = pd.read_csv(pv_csv_path)

# 2. Parse timestamps and compute period (1..96)
timestamp_col = pd.to_datetime(df["utc_timestamp"]).dt.tz_localize(None)
period_col = (timestamp_col.dt.hour * 4 + timestamp_col.dt.minute // 15) + 1

# 3. Column mapping (shorten names)
column_mapping = {
    "DE_KN_residential1_pv": "residential1",
    "DE_KN_residential3_pv": "residential3",
    "DE_KN_residential4_pv": "residential4",
    "DE_KN_residential6_pv": "residential6",
}

# 4. Convert cumulative meter readings to power in kW: (meter[t] - meter[t-1]) * 4
power_df = pd.DataFrame({
    "timestamp_utc": timestamp_col.astype(str),
    "period": period_col,
})

for raw_col, short_name in column_mapping.items():
    if raw_col in df.columns:
        # diff() * 4, negative artifacts clipped to 0, first value filled with 0
        diff_kw = (df[raw_col].diff() * 4).clip(lower=0.0).fillna(0.0)
        power_df[short_name] = diff_kw

# 5. Drop empty periods where all residential values are 0 or NaN if needed, 
# or sort and save directly
power_df = power_df.sort_values(["timestamp_utc", "period"]).reset_index(drop=True)

# 6. Save to staging SQLite DB
with sqlite3.connect(sqlite_path) as conn:
    power_df.to_sql("pv_gen", conn, if_exists="replace", index=False)

print(f"Wrote {len(power_df)} rows and {len(column_mapping)} households to {sqlite_path}")