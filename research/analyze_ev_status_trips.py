"""Summarize observed EV commute periods from the wide status tables."""

from __future__ import annotations

import argparse
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT_DIR / "sqlite" / "energy_v2.db"
PERIODS_PER_DAY = 96
MINUTES_PER_PERIOD = 15
EV_TABLES = ("ev1_status", "ev2_status")


@dataclass(frozen=True)
class Trip:
    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start + 1


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _load_trips(
    connection: sqlite3.Connection,
    table_name: str,
) -> tuple[dict[int, list[Trip]], Counter[int], int]:
    columns = [
        row[1]
        for row in connection.execute(
            f"PRAGMA table_info({_quote_identifier(table_name)})"
        )
    ]
    if not columns or columns[0] != "period":
        raise ValueError(f"{table_name} must have period as its first column")

    profile_columns = columns[1:]
    if not profile_columns:
        raise ValueError(f"{table_name} has no EV status profiles")

    selected_columns = ", ".join(_quote_identifier(column) for column in columns)
    rows = connection.execute(
        f"SELECT {selected_columns} FROM {_quote_identifier(table_name)} ORDER BY period"
    ).fetchall()
    periods = [row[0] for row in rows]
    if periods != list(range(1, PERIODS_PER_DAY + 1)):
        raise ValueError(
            f"{table_name} must contain periods 1..{PERIODS_PER_DAY}; "
            f"found {len(periods)} rows"
        )

    trips_by_ordinal: dict[int, list[Trip]] = defaultdict(list)
    trip_counts: Counter[int] = Counter()
    valid_statuses = {0, 1, 2}

    for profile_index, profile_column in enumerate(profile_columns, start=1):
        profile_trips: list[Trip] = []
        trip_start: int | None = None

        for period, row in enumerate(rows, start=1):
            status = row[profile_index]
            if status not in valid_statuses:
                raise ValueError(
                    f"{table_name}.{profile_column} has invalid status {status!r} "
                    f"at period {period}"
                )
            if status == 1 and trip_start is None:
                trip_start = period
            elif status != 1 and trip_start is not None:
                profile_trips.append(Trip(trip_start, period - 1))
                trip_start = None

        if trip_start is not None:
            profile_trips.append(Trip(trip_start, PERIODS_PER_DAY))

        trip_counts[len(profile_trips)] += 1
        for ordinal, trip in enumerate(profile_trips, start=1):
            trips_by_ordinal[ordinal].append(trip)

    return trips_by_ordinal, trip_counts, len(profile_columns)


def _print_report(
    table_name: str,
    trips_by_ordinal: dict[int, list[Trip]],
    trip_counts: Counter[int],
    profile_count: int,
) -> None:
    exact_two = trip_counts[2]
    fewer_than_two = sum(count for trips, count in trip_counts.items() if trips < 2)
    more_than_two = sum(count for trips, count in trip_counts.items() if trips > 2)
    total_trips = sum(len(trips) for trips in trips_by_ordinal.values())

    print(
        f"\n{table_name}: {profile_count} profiles, {total_trips} commute episodes; "
        f"profiles with exactly 2 episodes: {exact_two}, fewer: {fewer_than_two}, "
        f"more: {more_than_two}"
    )
    print(
        "trip | profiles | earliest start | latest end | max duration "
        "(steps / minutes)"
    )
    for ordinal, trips in sorted(trips_by_ordinal.items()):
        earliest_start = min(trip.start for trip in trips)
        latest_end = max(trip.end for trip in trips)
        max_duration = max(trip.duration for trip in trips)
        print(
            f"{ordinal:>4} | {len(trips):>8} | {earliest_start:>14} | "
            f"{latest_end:>10} | {max_duration:>4} / "
            f"{max_duration * MINUTES_PER_PERIOD}"
        )


def analyze(db_path: Path) -> None:
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")

    database_uri = f"{db_path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(database_uri, uri=True) as connection:
        available_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        missing_tables = [table for table in EV_TABLES if table not in available_tables]
        if missing_tables:
            raise ValueError(f"Missing EV status table(s): {', '.join(missing_tables)}")

        for table_name in EV_TABLES:
            trips_by_ordinal, trip_counts, profile_count = _load_trips(
                connection, table_name
            )
            _print_report(table_name, trips_by_ordinal, trip_counts, profile_count)

    print(
        "\nPeriods are 1-based. A commute episode is a consecutive run of status 1; "
        f"one period is {MINUTES_PER_PERIOD} minutes."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze commute episode times in EV status tables."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"SQLite database (default: {DEFAULT_DB})",
    )
    args = parser.parse_args()
    analyze(args.db)


if __name__ == "__main__":
    main()
