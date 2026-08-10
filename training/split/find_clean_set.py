"""Find and validate a clean 20-household training set.

Requirements:
- Exact category split: 12 PV+BESS, 4 PV-only, 4 no-PV/no-BESS.
- No two selected households may share an equivalent normalized base-load profile.
- No two selected PV households may share an equivalent normalized PV-generation profile.

Optional EV-state tie-breaker that we also satisfied:
- No two selected households share the same EV1 state profile.
- No two selected households share the same EV2 state profile.
- EV state is defined period-wise as: 1 - ev_at_home + ev_at_charging_station.

Approved candidate set:
[12, 29, 30, 38, 60, 62, 75, 93, 94, 112, 116, 121, 133, 154, 157, 171, 180, 204, 206, 217]
"""



from __future__ import annotations# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from importlib import import_module
from itertools import combinations
import random
from typing import TypedDict


sqlite_connection = import_module("src.sqlite_connection")
fetch_timeseries = sqlite_connection.fetch_timeseries
load_attribute = sqlite_connection.load_attribute
sqlite_cursor = sqlite_connection.sqlite_cursor

clean_set = [12, 29, 30, 38, 60, 62, 75, 93, 94, 112, 116, 121, 133, 154, 157, 171, 180, 204, 206, 217]

PLAYER_IDS = range(1, 251)
TARGET_COUNTS = {
    "no_pv_no_bess": 4,
    "pv_only": 4,
    "pv_bess": 12,
}
RNG_SEED = 42
MAX_ITERATIONS = 20000
PROFILE_TOLERANCE = 1e-5
TARGET_PRETTY_SETS = 25


class EvConflictScore(TypedDict):
    ev1_conflicts: int
    ev2_conflicts: int
    total_ev_conflicts: int
    ev1_groups: list[list[int]]
    ev2_groups: list[list[int]]


class CountMismatch(TypedDict):
    expected: int
    actual: int


class CandidateValidation(TypedDict):
    is_valid: bool
    selected_count: int
    category_counts: dict[str, int]
    wrong_counts: dict[str, CountMismatch]
    conflict_pairs: list[tuple[int, int]]
    ev_conflicts: EvConflictScore


class RankedPrettySet(TypedDict):
    selected_ids: list[int]
    ev_score: EvConflictScore


def _normalize_series(series: list[float]) -> list[float]:
    min_val = min(series)
    max_val = max(series)
    if max_val - min_val == 0:
        return [0.0 for _ in series]
    return [(value - min_val) / (max_val - min_val) for value in series]


def _equivalent_series(s1: list[float], s2: list[float], tolerance: float = PROFILE_TOLERANCE) -> bool:
    if len(s1) != len(s2):
        return False
    for idx in range(len(s1)):
        if abs(float(s1[idx]) - float(s2[idx])) > tolerance:
            return False
    return True


def _series_profile(player_id: int, table_name: str) -> list[float]:
    values = fetch_timeseries(sqlite_cursor, player_id=player_id, table_name=table_name)
    return _normalize_series(values)


def _player_category(player_id: int) -> str:
    has_pv = bool(load_attribute("player_pv_bess", player_id, "has_pv"))
    has_bess = bool(load_attribute("player_pv_bess", player_id, "has_bess"))

    if not has_pv:
        return "no_pv_no_bess"
    if not has_bess:
        return "pv_only"
    return "pv_bess"


def _ev_state_profile(player_id: int, ev_number: int) -> list[float]:
    home_values = fetch_timeseries(sqlite_cursor, player_id=player_id, table_name=f"ev{ev_number}_at_home")
    station_values = fetch_timeseries(
        sqlite_cursor,
        player_id=player_id,
        table_name=f"ev{ev_number}_at_charging_station",
    )
    return [
        1.0 - float(at_home) + float(at_station)
        for at_home, at_station in zip(home_values, station_values)
    ]


def _group_ids_by_equivalence(ids: list[int], profiles_by_id: dict[int, list[float]]) -> list[list[int]]:
    groups: list[list[int]] = []
    representative_profiles: list[list[float]] = []

    for player_id in sorted(ids):
        profile = profiles_by_id[player_id]
        assigned = False
        for group_idx, representative in enumerate(representative_profiles):
            if _equivalent_series(profile, representative):
                groups[group_idx].append(player_id)
                assigned = True
                break

        if not assigned:
            representative_profiles.append(profile)
            groups.append([player_id])

    return groups


def _build_groups() -> tuple[dict[int, str], list[list[int]], list[list[int]]]:
    categories: dict[int, str] = {}
    base_profiles: dict[int, list[float]] = {}
    pv_profiles: dict[int, list[float]] = {}

    for player_id in PLAYER_IDS:
        category = _player_category(player_id)
        categories[player_id] = category
        base_profiles[player_id] = _series_profile(player_id, "base_load")
        if category != "no_pv_no_bess":
            pv_profiles[player_id] = _series_profile(player_id, "pv_gen")

    base_groups = _group_ids_by_equivalence(list(categories.keys()), base_profiles)
    pv_groups = _group_ids_by_equivalence(list(pv_profiles.keys()), pv_profiles)

    return categories, base_groups, pv_groups


def _build_conflicts(
    categories: dict[int, str],
    base_groups: list[list[int]],
    pv_groups: list[list[int]],
) -> dict[int, set[int]]:
    conflicts = {player_id: set() for player_id in categories}

    for members in base_groups:
        for left_id, right_id in combinations(members, 2):
            conflicts[left_id].add(right_id)
            conflicts[right_id].add(left_id)

    for members in pv_groups:
        for left_id, right_id in combinations(members, 2):
            conflicts[left_id].add(right_id)
            conflicts[right_id].add(left_id)

    return conflicts


def _greedy_independent_set(player_ids: list[int], conflicts: dict[int, set[int]], rng: random.Random) -> list[int]:
    ordered_ids = player_ids[:]
    rng.shuffle(ordered_ids)

    selected_ids: list[int] = []
    blocked_ids: set[int] = set()
    for player_id in ordered_ids:
        if player_id in blocked_ids:
            continue
        selected_ids.append(player_id)
        blocked_ids.add(player_id)
        blocked_ids.update(conflicts[player_id])

    return selected_ids


def _extract_target_split(
    independent_ids: list[int],
    categories: dict[int, str],
    rng: random.Random,
) -> list[int] | None:
    pools: dict[str, list[int]] = {key: [] for key in TARGET_COUNTS}
    for player_id in independent_ids:
        pools[categories[player_id]].append(player_id)

    if any(len(pools[key]) < required for key, required in TARGET_COUNTS.items()):
        return None

    chosen_ids: list[int] = []
    for key, required in TARGET_COUNTS.items():
        pool = pools[key][:]
        rng.shuffle(pool)
        chosen_ids.extend(pool[:required])

    return sorted(chosen_ids)


def _count_conflict_violations(selected_ids: list[int], conflicts: dict[int, set[int]]) -> int:
    violations = 0
    selected_set = set(selected_ids)
    for player_id in selected_ids:
        for other_id in conflicts[player_id]:
            if other_id in selected_set and player_id < other_id:
                violations += 1
    return violations


def _count_categories(selected_ids: list[int], categories: dict[int, str]) -> dict[str, int]:
    counts = {key: 0 for key in TARGET_COUNTS}
    for player_id in selected_ids:
        counts[categories[player_id]] += 1
    return counts


def _ev_conflict_groups(selected_ids: list[int], ev_number: int) -> list[list[int]]:
    profiles_by_id = {player_id: _ev_state_profile(player_id, ev_number) for player_id in selected_ids}
    groups = _group_ids_by_equivalence(selected_ids, profiles_by_id)
    return [sorted(group) for group in groups if len(group) > 1]


def _count_group_conflicts(groups: list[list[int]]) -> int:
    return sum(len(group) * (len(group) - 1) // 2 for group in groups)


def score_ev_conflicts(selected_ids: list[int]) -> EvConflictScore:
    ev1_groups = _ev_conflict_groups(selected_ids, ev_number=1)
    ev2_groups = _ev_conflict_groups(selected_ids, ev_number=2)
    ev1_conflicts = _count_group_conflicts(ev1_groups)
    ev2_conflicts = _count_group_conflicts(ev2_groups)
    return {
        "ev1_conflicts": ev1_conflicts,
        "ev2_conflicts": ev2_conflicts,
        "total_ev_conflicts": ev1_conflicts + ev2_conflicts,
        "ev1_groups": ev1_groups,
        "ev2_groups": ev2_groups,
    }


def validate_candidate_set(candidate_ids: list[int]) -> CandidateValidation:
    categories, base_groups, pv_groups = _build_groups()
    conflicts = _build_conflicts(categories, base_groups, pv_groups)

    category_counts = _count_categories(candidate_ids, categories)
    conflict_pairs: list[tuple[int, int]] = []
    candidate_set = set(candidate_ids)
    for player_id in sorted(candidate_set):
        for other_id in sorted(conflicts[player_id]):
            if other_id in candidate_set and player_id < other_id:
                conflict_pairs.append((player_id, other_id))

    wrong_counts: dict[str, CountMismatch] = {
        category: {
            "expected": expected_count,
            "actual": category_counts[category],
        }
        for category, expected_count in TARGET_COUNTS.items()
        if category_counts[category] != expected_count
    }

    return {
        "is_valid": not wrong_counts and not conflict_pairs,
        "selected_count": len(candidate_ids),
        "category_counts": category_counts,
        "wrong_counts": wrong_counts,
        "conflict_pairs": conflict_pairs,
        "ev_conflicts": score_ev_conflicts(candidate_ids),
    }


def generate_pretty_sets(target_sets: int = TARGET_PRETTY_SETS) -> list[list[int]]:
    rng = random.Random(RNG_SEED)
    categories, base_groups, pv_groups = _build_groups()
    conflicts = _build_conflicts(categories, base_groups, pv_groups)
    player_ids = sorted(categories)

    seen_sets: set[tuple[int, ...]] = set()
    pretty_sets: list[list[int]] = []
    for _ in range(MAX_ITERATIONS):
        independent_ids = _greedy_independent_set(player_ids, conflicts, rng)
        chosen_ids = _extract_target_split(independent_ids, categories, rng)
        if chosen_ids is None:
            continue

        chosen_key = tuple(chosen_ids)
        if chosen_key in seen_sets:
            continue

        seen_sets.add(chosen_key)
        pretty_sets.append(chosen_ids)
        if len(pretty_sets) >= target_sets:
            break

    return pretty_sets


def pick_lowest_ev_conflict_set(pretty_sets: list[list[int]]) -> RankedPrettySet | None:
    ranked_sets: list[RankedPrettySet] = []
    for selected_ids in pretty_sets:
        ev_score = score_ev_conflicts(selected_ids)
        ranked_sets.append(
            {
                "selected_ids": selected_ids,
                "ev_score": ev_score,
            }
        )

    if not ranked_sets:
        return None

    ranked_sets.sort(
        key=lambda item: (
            item["ev_score"]["total_ev_conflicts"],
            item["ev_score"]["ev1_conflicts"],
            item["ev_score"]["ev2_conflicts"],
            item["selected_ids"],
        )
    )
    return ranked_sets[0]


def main() -> None:
    pretty_sets = generate_pretty_sets()
    best_choice = pick_lowest_ev_conflict_set(pretty_sets)

    print(f"pretty_set_count {len(pretty_sets)}")
    if best_choice is None:
        print("no_valid_set_found")
        return

    validation = validate_candidate_set(best_choice["selected_ids"])
    ev_score = best_choice["ev_score"]
    print(f"best_selected_ids {best_choice['selected_ids']}")
    print(f"best_selected_count {validation['selected_count']}")
    print(f"best_category_counts {validation['category_counts']}")
    print(f"best_base_pv_conflict_pairs {validation['conflict_pairs']}")
    print(f"best_ev1_conflicts {ev_score['ev1_conflicts']}")
    print(f"best_ev2_conflicts {ev_score['ev2_conflicts']}")
    print(f"best_total_ev_conflicts {ev_score['total_ev_conflicts']}")
    print(f"best_ev1_groups {ev_score['ev1_groups']}")
    print(f"best_ev2_groups {ev_score['ev2_groups']}")


if __name__ == "__main__":
    main()
    candidate_set = [12, 29, 30, 38, 60, 62, 75, 93, 94, 112, 116, 121, 133, 154, 157, 171, 180, 204, 206, 217, 201]
    validation = validate_candidate_set(candidate_set)
    print(f"candidate_set_valid {validation['is_valid']}")
    print(f"candidate_set_count {validation['selected_count']}")
    print(f"candidate_set_category_counts {validation['category_counts']}")
    print(f"candidate_set_wrong_counts {validation['wrong_counts']}")
    print(f"candidate_set_conflict_pairs {validation['conflict_pairs']}")
    print(f"candidate_set_ev1_conflicts {validation['ev_conflicts']['ev1_conflicts']}")
    print(f"candidate_set_ev2_conflicts {validation['ev_conflicts']['ev2_conflicts']}")
    print(f"candidate_set_total_ev_conflicts {validation['ev_conflicts']['total_ev_conflicts']}")