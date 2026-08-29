import sqlite3
import pandas as pd
import numpy as np

# 1. ch_smart_meters.db
print("=== Analyzing ch_smart_meters.db 'load' table ===")
conn_ch = sqlite3.connect('sqlite/ch_smart_meters.db')
cursor_ch = conn_ch.cursor()
cursor_ch.execute("PRAGMA table_info(load)")
cols_ch = [row[1] for row in cursor_ch.fetchall()]
player_cols_ch = [c for c in cols_ch if c not in ('timestamp_utc', 'period')]
print(f"Found {len(player_cols_ch)} player columns out of {len(cols_ch)} columns.")

# Read load table
df_ch = pd.read_sql_query("SELECT * FROM load", conn_ch)
# Melt or stack player columns and convert to float
ch_series = df_ch[player_cols_ch].stack().dropna().astype(float)
conn_ch.close()

# 2. energy.db
print("\n=== Analyzing energy.db 'base_load' table ===")
conn_en = sqlite3.connect('sqlite/energy.db')
cursor_en = conn_en.cursor()
cursor_en.execute("PRAGMA table_info(base_load)")
cols_en = [row[1] for row in cursor_en.fetchall()]
non_player_cols_en = [c for c in cols_en if 'utc' in c.lower() or 'period' in c.lower() or c.lower() in ('date', 'time', 'timestamp')]
player_cols_en = [c for c in cols_en if c not in non_player_cols_en]
print(f"Non-player columns in energy.base_load: {non_player_cols_en}")
print(f"Found {len(player_cols_en)} player columns out of {len(cols_en)} columns.")

df_en = pd.read_sql_query("SELECT * FROM base_load", conn_en)
en_series = df_en[player_cols_en].stack().dropna().astype(float)
conn_en.close()

# 3. Stats helper
def print_stats(name, series):
    print(f"\n--- Statistics for {name} ---")
    qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    quantiles = series.quantile(qs)
    
    print(f"Count:      {series.size:,}")
    print(f"Mean:       {series.mean():.6f}")
    print(f"Std:        {series.std():.6f}")
    print(f"Min:        {series.min():.6f}")
    for q in qs:
        print(f"Quantile {q}: {quantiles[q]:.6f}")
    print(f"Max:        {series.max():.6f}")
    
    print(f"\nTop 5 Peak Values for {name}:")
    top5 = series.nlargest(5)
    for idx, val in top5.items():
        print(f"  Index {idx}: {val:.6f}")

print_stats("ch_smart_meters.load", ch_series)
print_stats("energy.base_load", en_series)
