# ok again
# we want to find a set of ids where no pv_gen/base_load/ev_status profile is equivalent.7
# this also resulted in 20 ids so the 20 are fine


# paste this to enable src. imports
from pathlib import Path
import sys
from itertools import combinations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))


from src.sqlite_connection import fetch_timeseries, sqlite_cursor


def _normalize_series(series: list[float]) -> list[float]:
    if not series:
        return []

    min_val = min(series)
    max_val = max(series)
    if max_val - min_val == 0:
        return [0.0 for _ in series]
    return [(value - min_val) / (max_val - min_val) for value in series]


def _equivalent_series(s1: list[float], s2: list[float], tolerance: float = 1e-5) -> bool:
    if len(s1) != len(s2):
        return False

    for t in range(len(s1)):
        if abs(s1[t] - s2[t]) > tolerance:
            return False

    return True

class _UnionFind:
    def __init__(self, ids: list[int]):
        self.parent = {node: node for node in ids}
        self.rank = {node: 0 for node in ids}

    def find(self, node: int) -> int:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return

        rank_a = self.rank[root_a]
        rank_b = self.rank[root_b]
        if rank_a < rank_b:
            self.parent[root_a] = root_b
            return
        if rank_a > rank_b:
            self.parent[root_b] = root_a
            return

        self.parent[root_b] = root_a
        self.rank[root_a] += 1


def _find_equal_groups(key: str, normalized_profiles: dict, tolerance: float = 1e-5) -> dict[int, list[int]]:
    ids = sorted(normalized_profiles[key].keys())
    union_find = _UnionFind(ids)

    for left_pos, left_id in enumerate(ids):
        s1 = normalized_profiles[key][left_id]
        for right_id in ids[left_pos + 1 :]:
            s2 = normalized_profiles[key][right_id]
            if _equivalent_series(s1, s2, tolerance=tolerance):
                union_find.union(left_id, right_id)

    components = {}
    for player_id in ids:
        root = union_find.find(player_id)
        components.setdefault(root, []).append(player_id)

    equal_groups = {}
    for grouped_ids in components.values():
        if len(grouped_ids) > 1:
            grouped_ids = sorted(grouped_ids)
            equal_groups[grouped_ids[0]] = grouped_ids

    return equal_groups


def _build_conflict_pairs(equal_groups_by_key: dict[str, dict[int, list[int]]]) -> set[tuple[int, int]]:
    conflict_pairs = set()
    for equal_groups in equal_groups_by_key.values():
        for grouped_ids in equal_groups.values():
            for id_a, id_b in combinations(sorted(grouped_ids), 2):
                conflict_pairs.add((id_a, id_b))
    return conflict_pairs


def _solve_max_clean_set(all_ids: list[int], conflict_pairs: set[tuple[int, int]]) -> tuple[list[int], object | None]:
    if not conflict_pairs:
        return sorted(all_ids), None

    ordered_ids = sorted(all_ids)
    n = len(ordered_ids)
    id_to_index = {player_id: idx for idx, player_id in enumerate(ordered_ids)}

    c = -np.ones(n, dtype=float)
    integrality = np.ones(n, dtype=int)
    bounds = Bounds(np.zeros(n), np.ones(n))

    pairs = sorted(conflict_pairs)
    rows = []
    cols = []
    data = []
    for row_idx, (id_a, id_b) in enumerate(pairs):
        rows.extend([row_idx, row_idx])
        cols.extend([id_to_index[id_a], id_to_index[id_b]])
        data.extend([1.0, 1.0])

    A = coo_matrix((data, (rows, cols)), shape=(len(pairs), n)).tocsr()
    lc = LinearConstraint(
        A,
        -np.inf * np.ones(len(pairs), dtype=float),
        np.ones(len(pairs), dtype=float),
    )

    result = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=[lc],
    )

    if not result.success:
        raise RuntimeError(f"MILP failed: {result.message}")

    selected = []
    for idx, value in enumerate(result.x):
        if value >= 0.5:
            selected.append(ordered_ids[idx])

    return selected, result


def _validate_solution(selected_ids: list[int], conflict_pairs: set[tuple[int, int]]) -> list[tuple[int, int]]:
    selected = set(selected_ids)
    violations = []
    for id_a, id_b in sorted(conflict_pairs):
        if id_a in selected and id_b in selected:
            violations.append((id_a, id_b))
    return violations



raw_profiles = {
    "base_load": {},
    "pv_gen": {},
    "ev1_status": {},
    "ev2_status": {}
}

normalized_profiles = {
    "base_load": {},
    "pv_gen": {},
    "ev1_status": {},
    "ev2_status": {}
}

for id in range(1, 251):
    for key in raw_profiles.keys():
        profile = fetch_timeseries(sqlite_cursor, id, key)
        raw_profiles[key][id] = profile
        normalized_profiles[key][id] = _normalize_series(profile)


all_ids = list(range(1, 251))
equal_groups_by_key = {}
for key in normalized_profiles.keys():
    equal_groups_by_key[key] = _find_equal_groups(key, normalized_profiles)

conflict_pairs = _build_conflict_pairs(equal_groups_by_key)
selected_clean_ids, solver_result = _solve_max_clean_set(all_ids, conflict_pairs)
violations = _validate_solution(selected_clean_ids, conflict_pairs)


print("Profile keys:", list(normalized_profiles.keys()))
for key in normalized_profiles.keys():
    duplicate_count = sum(len(group) for group in equal_groups_by_key[key].values())
    print(f"{key}: ids in duplicate groups = {duplicate_count}")
    print(f"{key}: equal groups = {equal_groups_by_key[key]}")

print("\nConflict edges:", len(conflict_pairs))
print("MILP selected count:", len(selected_clean_ids))
print("MILP selected ids:", selected_clean_ids)

if solver_result is not None:
    print("MILP status:", solver_result.status)
    print("MILP objective (min -sum(x)):", float(solver_result.fun))

if violations:
    print("Constraint violations found:", violations)
else:
    print("Constraint check: OK (no equivalent pair selected together)")

# [9, 14, 21, 50, 54, 75, 94, 98, 102, 127, 129, 146, 152, 166, 177, 200, 201, 202, 231, 243]

# base load for 29
s1 = normalized_profiles["pv_gen"][29]
s2 = normalized_profiles["pv_gen"][201]


equiv_test = _equivalent_series(s1, s2)

print(f"pv_gen  29 vs 201 equivalent: {equiv_test}")