"""Compare maximum base-load distributions from the two SQLite datasets."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ENERGY_DB = ROOT / "sqlite" / "energy.db"
CH_DB = ROOT / "sqlite" / "ch_smart_meters.db"
OUTPUT_DIR = Path(__file__).resolve().parent
BUCKET_WIDTH = 0.1


def _player_columns(connection: sqlite3.Connection, table_name: str) -> list[str]:
    columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{table_name}")')]
    metadata = {"timestamp_utc", "period", "date", "time", "timestamp"}
    return [column for column in columns if column.lower() not in metadata]


def load_energy_maxima() -> pd.Series:
    with sqlite3.connect(ENERGY_DB) as connection:
        player_columns = _player_columns(connection, "base_load")
        frame = pd.read_sql_query('SELECT * FROM "base_load"', connection)

    return frame[player_columns].apply(pd.to_numeric, errors="coerce").max(axis=0).dropna()


def load_ch_daily_maxima() -> pd.Series:
    with sqlite3.connect(CH_DB) as connection:
        player_columns = _player_columns(connection, "load")
        frame = pd.read_sql_query('SELECT * FROM "load"', connection)

    frame["date"] = pd.to_datetime(frame["timestamp_utc"], errors="coerce").dt.date
    values = frame[player_columns].apply(pd.to_numeric, errors="coerce")
    daily_maxima = values.assign(date=frame["date"]).groupby("date").max()
    return daily_maxima.stack().dropna()


def _bucket_values(values: pd.Series) -> pd.Series:
    """Return the upper edge of each 0.1-wide bucket (e.g. 1.1, 1.2, ...)."""
    return ((values / BUCKET_WIDTH).apply(lambda value: int(value + 0.999999)) * BUCKET_WIDTH).round(1)


def _histogram(
    values: pd.Series,
    title: str,
    output_path: Path,
    color: str,
) -> None:
    buckets = _bucket_values(values)
    counts = buckets.value_counts().sort_index()
    width = BUCKET_WIDTH * 0.9

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(counts.index, counts.values, width=width, color=color, edgecolor="white", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("Maximum load bucket (kW)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _overlay_histogram(energy_values: pd.Series, ch_values: pd.Series, output_path: Path) -> None:
    energy_buckets = _bucket_values(energy_values)
    ch_buckets = _bucket_values(ch_values)
    all_buckets = sorted(set(energy_buckets) | set(ch_buckets))
    energy_counts = energy_buckets.value_counts().reindex(all_buckets, fill_value=0)
    ch_counts = ch_buckets.value_counts().reindex(all_buckets, fill_value=0)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(all_buckets, energy_counts, width=BUCKET_WIDTH * 0.42, label="energy.db base_load", alpha=0.75)
    ax.bar(
        [bucket + BUCKET_WIDTH * 0.42 for bucket in all_buckets],
        ch_counts,
        width=BUCKET_WIDTH * 0.42,
        label="ch_smart_meters.db daily maxima",
        alpha=0.75,
    )
    ax.set_title("Maximum load distributions")
    ax.set_xlabel("Maximum load bucket (kW)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    if not ENERGY_DB.exists() or not CH_DB.exists():
        missing = [str(path) for path in (ENERGY_DB, CH_DB) if not path.exists()]
        raise FileNotFoundError(f"Missing database(s): {', '.join(missing)}")

    energy_maxima = load_energy_maxima()
    ch_daily_maxima = load_ch_daily_maxima()
    if energy_maxima.empty or ch_daily_maxima.empty:
        raise ValueError("One of the maximum-load datasets is empty.")

    _histogram(
        energy_maxima,
        "Maximum base load per player: energy.db",
        OUTPUT_DIR / "max_load_histogram_energy_db.png",
        "#1f77b4",
    )
    _histogram(
        ch_daily_maxima,
        "Daily maximum load per player: ch_smart_meters.db",
        OUTPUT_DIR / "max_load_histogram_ch_smart_meters.png",
        "#ff7f0e",
    )
    _overlay_histogram(
        energy_maxima,
        ch_daily_maxima,
        OUTPUT_DIR / "max_load_histogram_overlay.png",
    )

    print(f"energy.db maxima: {len(energy_maxima):,} players")
    print(f"CH daily maxima: {len(ch_daily_maxima):,} player-days")
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()