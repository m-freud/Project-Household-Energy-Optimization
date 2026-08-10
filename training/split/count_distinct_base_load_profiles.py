"""Count distinct normalized base-load profile classes.

By default this uses tolerance-based equality (same rule as _equivalent_series):
two normalized profiles are equivalent if every timestep differs by <= tolerance.

Optional mode "rounded" is available to reproduce signature grouping by
rounding each normalized value to a fixed number of digits.

Examples:
  python training/split/count_distinct_base_load_profiles.py
  python training/split/count_distinct_base_load_profiles.py --show-groups
  python training/split/count_distinct_base_load_profiles.py --mode rounded --round-digits 6
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import sys


# Enable imports from src/
cwd = Path.cwd().resolve()
repo_root = next((p for p in [cwd, *cwd.parents] if (p / "src").exists()), cwd)
sys.path.insert(0, str(repo_root))

from src.sqlite_connection import fetch_timeseries, sqlite_cursor


def _normalize_series(series: list[float]) -> list[float]:
    if not series:
        return []

    min_val = min(series)
    max_val = max(series)
    span = max_val - min_val
    if span == 0:
        return [0.0 for _ in series]

    return [(value - min_val) / span for value in series]


def _equivalent_series(s1: list[float], s2: list[float], tolerance: float) -> bool:
    if len(s1) != len(s2):
        return False
    for idx in range(len(s1)):
        if abs(float(s1[idx]) - float(s2[idx])) > tolerance:
            return False
    return True


def _count_distinct_tolerance(
    player_ids: list[int],
    tolerance: float,
) -> list[list[int]]:
    representative_profiles: list[list[float]] = []
    groups: list[list[int]] = []

    for player_id in player_ids:
        profile = _normalize_series(fetch_timeseries(sqlite_cursor, player_id, "base_load"))

        matched_group_idx = None
        for group_idx, representative in enumerate(representative_profiles):
            if _equivalent_series(profile, representative, tolerance):
                matched_group_idx = group_idx
                break

        if matched_group_idx is None:
            representative_profiles.append(profile)
            groups.append([player_id])
        else:
            groups[matched_group_idx].append(player_id)

    return groups


def _count_distinct_rounded(
    player_ids: list[int],
    round_digits: int,
) -> list[list[int]]:
    grouped: dict[tuple[float, ...], list[int]] = defaultdict(list)

    for player_id in player_ids:
        normalized = _normalize_series(fetch_timeseries(sqlite_cursor, player_id, "base_load"))
        signature = tuple(round(float(value), round_digits) for value in normalized)
        grouped[signature].append(player_id)

    return [sorted(ids) for ids in grouped.values()]


def main() -> None:
    parser = ArgumentParser(description="Count distinct normalized base-load profile classes.")
    parser.add_argument("--start-id", type=int, default=1, help="First player id (inclusive).")
    parser.add_argument("--end-id", type=int, default=250, help="Last player id (inclusive).")
    parser.add_argument(
        "--mode",
        choices=["tolerance", "rounded"],
        default="tolerance",
        help="Equivalence mode used to define a distinct profile class.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Absolute tolerance used when --mode tolerance.",
    )
    parser.add_argument(
        "--round-digits",
        type=int,
        default=6,
        help="Digits used when --mode rounded.",
    )
    parser.add_argument(
        "--show-groups",
        action="store_true",
        help="Print all distinct groups and their member ids.",
    )
    args = parser.parse_args()

    player_ids = list(range(int(args.start_id), int(args.end_id) + 1))
    if args.mode == "tolerance":
        groups = _count_distinct_tolerance(player_ids, tolerance=float(args.tolerance))
    else:
        groups = _count_distinct_rounded(player_ids, round_digits=int(args.round_digits))

    groups = [sorted(group) for group in groups]
    groups.sort(key=lambda group: (group[0], len(group)))

    print(f"mode {args.mode}")
    if args.mode == "tolerance":
        print(f"tolerance {float(args.tolerance)}")
    else:
        print(f"round_digits {int(args.round_digits)}")

    print(f"id_range {args.start_id}-{args.end_id}")
    print(f"distinct_base_load_profile_count {len(groups)}")

    if args.show_groups:
        print(f"groups {groups}")


if __name__ == "__main__":
    main()
