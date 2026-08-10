"""Find metric-distinct IDs outside CLEAN_SET.

This script selects outside IDs whose metric profile signature is not present in
Config.CLEAN_SET. For each unseen signature, it keeps one representative ID
(smallest ID for deterministic output).

It also prints fold-wise training IDs:
- train_ids[label] = Config.CLEAN_COMPLEMENTS[label] + extra_ids

Usage examples:
  python training/split/find_metric_distinct_ids.py --metric pv_gen
  python training/split/find_metric_distinct_ids.py --metric ev1_status
  python training/split/find_metric_distinct_ids.py --metric ev_status
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import sys


# Enable imports from src/
cwd = Path.cwd().resolve()
repo_root = next((p for p in [cwd, *cwd.parents] if (p / "src").exists()), cwd)
sys.path.insert(0, str(repo_root))

from src.config import Config
from src.sqlite_connection import fetch_timeseries, sqlite_cursor


MetricName = str


def _normalize_series(series: list[float]) -> list[float]:
    if not series:
        return []

    min_val = min(series)
    max_val = max(series)
    span = max_val - min_val
    if span == 0:
        return [0.0 for _ in series]
    return [(value - min_val) / span for value in series]


def _equivalent_series(s1: list[float], s2: list[float], tolerance: float = 1e-5) -> bool:
    if len(s1) != len(s2):
        return False
    for idx in range(len(s1)):
        if abs(float(s1[idx]) - float(s2[idx])) > tolerance:
            return False
    return True


def _metric_profile(player_id: int, metric: MetricName) -> list[float]:
    if metric == "ev_status":
        ev1 = fetch_timeseries(sqlite_cursor, player_id, "ev1_status")
        ev2 = fetch_timeseries(sqlite_cursor, player_id, "ev2_status")
        p1 = _normalize_series(ev1)
        p2 = _normalize_series(ev2)
        return p1 + p2

    profile = fetch_timeseries(sqlite_cursor, player_id, metric)
    return _normalize_series(profile)


def _collect_player_ids(start_id: int, end_id: int) -> list[int]:
    return list(range(start_id, end_id + 1))


def build_extra_ids_for_metric(
    metric: MetricName,
    all_ids: list[int],
    clean_set: list[int],
    tolerance: float,
) -> tuple[list[int], int, list[list[int]]]:
    clean_representative_profiles: list[list[float]] = []

    for player_id in sorted(clean_set):
        profile = _metric_profile(player_id, metric)
        if any(
            _equivalent_series(profile, existing_profile, tolerance=tolerance)
            for existing_profile in clean_representative_profiles
        ):
            continue
        clean_representative_profiles.append(profile)

    outside_ids = sorted(player_id for player_id in all_ids if player_id not in set(clean_set))

    extras: list[int] = []
    uncovered_groups: list[list[int]] = []
    selected_extra_profiles: list[list[float]] = []

    for player_id in outside_ids:
        profile = _metric_profile(player_id, metric)

        if any(
            _equivalent_series(profile, clean_profile, tolerance=tolerance)
            for clean_profile in clean_representative_profiles
        ):
            continue

        if any(
            _equivalent_series(profile, existing_profile, tolerance=tolerance)
            for existing_profile in selected_extra_profiles
        ):
            for group_idx, existing_profile in enumerate(selected_extra_profiles):
                if _equivalent_series(profile, existing_profile, tolerance=tolerance):
                    uncovered_groups[group_idx].append(player_id)
                    break
            continue

        selected_extra_profiles.append(profile)
        extras.append(player_id)
        uncovered_groups.append([player_id])

    return extras, len(clean_representative_profiles), uncovered_groups


def build_fold_training_sets(extra_ids: list[int]) -> dict[str, list[int]]:
    training_sets: dict[str, list[int]] = {}
    for label, clean_complement in Config.CLEAN_COMPLEMENTS.items():
        training_sets[label] = sorted(clean_complement + extra_ids)
    return training_sets


def main() -> None:
    parser = ArgumentParser(description="Find outside IDs with metric signatures not present in CLEAN_SET.")
    parser.add_argument(
        "--metric",
        required=True,
        choices=["base_load", "pv_gen", "ev1_status", "ev2_status", "ev_status"],
        help="Metric table name (or combined ev_status = ev1_status + ev2_status signatures).",
    )
    parser.add_argument("--start-id", type=int, default=1, help="First player ID to scan (inclusive).")
    parser.add_argument("--end-id", type=int, default=250, help="Last player ID to scan (inclusive).")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Absolute tolerance for profile equivalence after normalization.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional output file path for JSON payload.",
    )
    parser.add_argument(
        "--show-groups",
        action="store_true",
        help="Print uncovered outside groups (IDs sharing each new profile).",
    )
    args = parser.parse_args()

    all_ids = _collect_player_ids(args.start_id, args.end_id)
    clean_set = sorted(Config.CLEAN_SET)

    extra_ids, clean_signature_count, uncovered_groups = build_extra_ids_for_metric(
        metric=args.metric,
        all_ids=all_ids,
        clean_set=clean_set,
        tolerance=float(args.tolerance),
    )

    fold_training_sets = build_fold_training_sets(extra_ids)

    payload = {
        "metric": args.metric,
        "clean_set_size": len(clean_set),
        "clean_set": clean_set,
        "clean_signature_count": clean_signature_count,
        "total_signature_count": clean_signature_count + len(uncovered_groups),
        "outside_extra_id_count": len(extra_ids),
        "outside_extra_ids": extra_ids,
        "outside_uncovered_groups": uncovered_groups,
        "fold_training_sets": fold_training_sets,
    }

    print(f"metric {payload['metric']}")
    print(f"clean_set_size {payload['clean_set_size']}")
    print(f"clean_signature_count {payload['clean_signature_count']}")
    print(f"total_signature_count {payload['total_signature_count']}")
    print(f"outside_extra_id_count {payload['outside_extra_id_count']}")
    print(f"outside_extra_ids {payload['outside_extra_ids']}")
    if args.show_groups:
        print(f"outside_uncovered_groups {payload['outside_uncovered_groups']}")
    for label in sorted(fold_training_sets):
        print(f"fold_{label}_train_count {len(fold_training_sets[label])}")
        print(f"fold_{label}_train_ids {fold_training_sets[label]}")

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"json_written {output_path}")


if __name__ == "__main__":
    main()
