"""
Calculate metadata tables for the original data in energy.db.

Same metrics as the data/*/create_*metadata*.py scripts, but the original
data covers a single day per player, so there is no `date` column.

Output tables (in energy.db):
    base_load_metadata   from base_load
    pv_gen_metadata      from pv_gen
    ev1_metadata         from ev1, ev1_status
    ev2_metadata         from ev2, ev2_status
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
from src.runtime_config import RuntimeConfig


PERIOD_HOURS = 0.25
PERIODS_PER_DAY = 96

# 06:00 - 11:45
MORNING_PERIODS = range(25, 49)
# 17:00 - 22:45
EVENING_PERIODS = range(69, 93)

# Generation below this threshold is treated as zero.
ACTIVE_THRESHOLD_KW = 0.05


def read_wide(conn, table):
    df = pd.read_sql(f"SELECT * FROM {table}", conn).sort_values("period")
    return df.reset_index(drop=True)


def player_columns(df):
    return [col for col in df.columns if col != "period"]


# ---------------------------------------------------------------------------
# Base load
# ---------------------------------------------------------------------------

def build_base_load_metadata(conn):
    df = read_wide(conn, "base_load")
    rows = []

    for player_id in player_columns(df):
        values = df[player_id].dropna().astype(float)
        if len(values) != PERIODS_PER_DAY:
            continue

        max_load = values.max()
        mean_load = values.mean()

        rows.append(
            {
                "player_id": int(player_id),
                "total_energy": values.sum() * PERIOD_HOURS,
                "max_load": max_load,
                "std_load": values.std(),
                "load_factor": mean_load / max_load if max_load > 0 else 0.0,
                "peak_period": int(df.loc[values.idxmax(), "period"]),
                "morning_mean": df.loc[df["period"].isin(MORNING_PERIODS), player_id].mean(),
                "evening_mean": df.loc[df["period"].isin(EVENING_PERIODS), player_id].mean(),
                "ramp_std": values.diff().dropna().std(),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PV generation
# ---------------------------------------------------------------------------

def build_pv_gen_metadata(conn):
    df = read_wide(conn, "pv_gen")
    rows = []

    for player_id in player_columns(df):
        values = df[player_id].dropna().astype(float)
        if len(values) != PERIODS_PER_DAY:
            continue

        max_generation = values.max()

        # Skip systems without meaningful PV generation
        if max_generation <= ACTIVE_THRESHOLD_KW:
            continue

        active_mask = values > ACTIVE_THRESHOLD_KW
        active_indices = values.index[active_mask]
        ramps = values.diff().dropna()

        rows.append(
            {
                "player_id": int(player_id),
                "total_energy": values.sum() * PERIOD_HOURS,
                "max_generation": max_generation,
                "std_generation": values.std(),
                "generation_factor": values.mean() / max_generation,
                "peak_period": int(df.loc[values.idxmax(), "period"]),
                "generation_start": int(df.loc[active_indices[0], "period"]),
                "generation_end": int(df.loc[active_indices[-1], "period"]),
                "active_periods": int(active_mask.sum()),
                "ramp_std": ramps.std(),
                "max_ramp": ramps.abs().max(),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# EV
# ---------------------------------------------------------------------------

def driving_blocks(periods):
    blocks = []
    for period in periods:
        if not blocks or period != blocks[-1][-1] + 1:
            blocks.append([period])
        else:
            blocks[-1].append(period)
    return blocks


def build_ev_metadata(conn, ev_num, detailed):
    status_df = read_wide(conn, f"ev{ev_num}_status")
    ev_df = pd.read_sql(f"SELECT player_id, distance_km FROM ev{ev_num}", conn)
    distances = dict(zip(ev_df["player_id"].astype(int), ev_df["distance_km"]))

    rows = []

    for player_id in player_columns(status_df):
        status = status_df[player_id].astype(int)
        if len(status) != PERIODS_PER_DAY:
            continue

        blocks = driving_blocks(status_df.loc[status == 1, "period"].tolist())
        trip_count = len(blocks)

        # The original data only has one total distance per player
        total_distance = distances.get(int(player_id))
        trip_distance = (
            total_distance / trip_count if trip_count and total_distance is not None else None
        )

        away_periods = int((status != 0).sum())

        row = {
            "player_id": int(player_id),
            "trip_count": trip_count,
            "total_distance": total_distance,
            "mean_trip_distance": trip_distance,
            "max_trip_distance": trip_distance,
        }

        if detailed:
            row.update(
                {
                    "outbound_distance": trip_distance if trip_count >= 1 else None,
                    "return_distance": trip_distance if trip_count >= 2 else None,
                    "outbound_start": int(blocks[0][0]) if trip_count >= 1 else None,
                    "outbound_end": int(blocks[0][-1]) if trip_count >= 1 else None,
                    "return_start": int(blocks[1][0]) if trip_count >= 2 else None,
                    "return_end": int(blocks[1][-1]) if trip_count >= 2 else None,
                }
            )

        row.update(
            {
                "first_departure": int(blocks[0][0]) if trip_count else None,
                "last_arrival": int(blocks[-1][-1]) if trip_count else None,
                "driving_periods": int((status == 1).sum()),
            }
        )

        if detailed:
            row["work_periods"] = int((status == 2).sum())

        row.update(
            {
                "away_periods": away_periods,
                "home_periods": int((status == 0).sum()),
                "away_fraction": away_periods / PERIODS_PER_DAY,
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save(conn, df, table, unique_col, index_cols):
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_{unique_col} ON {table} ({unique_col})")
    for col in index_cols:
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON {table} ({col})")
    print(f"{table}: {len(df)} rows")


if __name__ == "__main__":
    with sqlite3.connect(RuntimeConfig.SQLITE_PATH) as conn:
        save(conn, build_base_load_metadata(conn), "base_load_metadata", "player_id", ["max_load", "total_energy"])
        save(conn, build_pv_gen_metadata(conn), "pv_gen_metadata", "player_id", ["max_generation", "total_energy"])
        # EV1 is the commute profile (detailed), EV2 the general mobility profile
        save(conn, build_ev_metadata(conn, 1, detailed=True), "ev1_metadata", "player_id", ["total_distance"])
        save(conn, build_ev_metadata(conn, 2, detailed=False), "ev2_metadata", "player_id", ["total_distance", "trip_count"])
