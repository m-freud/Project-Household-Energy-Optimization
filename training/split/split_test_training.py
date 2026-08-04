import json
import sqlite3
from collections import defaultdict
from pathlib import Path
import sys

import pandas as pd


repo_root = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents] if (p / "src").exists()), None)
if repo_root is None:
    raise RuntimeError("Could not locate repository root containing 'src'.")
sys.path.insert(0, str(repo_root))

from src.config import Config


PROFILE_TABLES = ("base_load", "pv_gen")
EV_KEYS = ("ev1", "ev2")


class UnionFind:
    def __init__(self, items: list[int]):
        self.parent = {item: item for item in items}
        self.size = {item: 1 for item in items}

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return

        if self.size[root_left] < self.size[root_right]:
            root_left, root_right = root_right, root_left

        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]


def _load_wide_table(table_name: str) -> pd.DataFrame:
    with sqlite3.connect(Config.SQLITE_PATH) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY period", conn)


def _household_columns(df: pd.DataFrame) -> list[str]:
    return sorted([column for column in df.columns if str(column).isdigit()], key=int)


def _profile_signature(values: pd.Series) -> tuple:
    return tuple(pd.to_numeric(values, errors="coerce").fillna(0).tolist())


def _load_generic_profiles(table_name: str) -> dict[int, tuple]:
    df = _load_wide_table(table_name)
    return {int(column): _profile_signature(df[column]) for column in _household_columns(df)}


def _load_ev_status_profiles(ev_key: str) -> dict[int, tuple]:
    at_home = _load_wide_table(f"{ev_key}_at_home")
    at_station = _load_wide_table(f"{ev_key}_at_charging_station")
    common_columns = sorted(set(_household_columns(at_home)).intersection(_household_columns(at_station)), key=int)

    profiles: dict[int, tuple] = {}
    for column in common_columns:
        at_home_values = pd.to_numeric(at_home[column], errors="coerce").fillna(0).astype(int)
        at_station_values = pd.to_numeric(at_station[column], errors="coerce").fillna(0).astype(int)
        status_values = 1 - at_home_values + at_station_values
        profiles[int(column)] = tuple(status_values.tolist())

    return profiles


def _collect_all_profiles() -> dict[str, dict[int, tuple]]:
    profiles: dict[str, dict[int, tuple]] = {}
    for table_name in PROFILE_TABLES:
        profiles[table_name] = _load_generic_profiles(table_name)
    for ev_key in EV_KEYS:
        profiles[f"{ev_key}_status"] = _load_ev_status_profiles(ev_key)
    return profiles


def _build_duplicate_groups(profiles_by_name: dict[str, dict[int, tuple]]) -> tuple[list[int], dict[str, list[list[int]]]]:
    all_households = sorted({household_id for profiles in profiles_by_name.values() for household_id in profiles})
    union_find = UnionFind(all_households)
    duplicate_groups_by_profile: dict[str, list[list[int]]] = {}

    for profile_name, profile_map in profiles_by_name.items():
        grouped: dict[tuple, list[int]] = defaultdict(list)
        for household_id, signature in profile_map.items():
            grouped[signature].append(household_id)

        duplicate_groups = [sorted(group) for group in grouped.values() if len(group) > 1]
        duplicate_groups_by_profile[profile_name] = sorted(duplicate_groups, key=lambda group: (group[0], len(group)))

        for group in duplicate_groups:
            anchor = group[0]
            for household_id in group[1:]:
                union_find.union(anchor, household_id)

    merged_groups: dict[int, list[int]] = defaultdict(list)
    for household_id in all_households:
        merged_groups[union_find.find(household_id)].append(household_id)

    household_groups = sorted((sorted(group) for group in merged_groups.values()), key=lambda group: (-len(group), group[0]))
    return all_households, {"merged": household_groups, **duplicate_groups_by_profile}


def _split_groups(household_groups: list[list[int]], target_train_size: int) -> tuple[list[int], list[int]]:
    training: list[int] = []
    testing: list[int] = []

    for group in household_groups:
        train_if_added = len(training) + len(group)
        keep_in_train = abs(target_train_size - train_if_added) <= abs(target_train_size - len(training))
        if len(training) < target_train_size and keep_in_train:
            training.extend(group)
        else:
            testing.extend(group)

    if not testing:
        last_group_size = len(household_groups[-1])
        testing = training[-last_group_size:]
        training = training[:-last_group_size]

    return sorted(training), sorted(testing)


def clean_split(target_train_size: int = 150) -> dict[str, list[int]]:
    '''
    Split households into training and testing sets without cross-duplicated profiles.

    Households are grouped together when they share an identical profile in any of:
    - base_load
    - pv_gen
    - ev1_status (derived 0/1/2)
    - ev2_status (derived 0/1/2)
    '''
    _, groups_by_profile = _build_duplicate_groups(_collect_all_profiles())
    training, testing = _split_groups(groups_by_profile["merged"], target_train_size=target_train_size)
    return {
        "training": training,
        "testing": testing,
    }


def _print_duplicate_report(groups_by_profile: dict[str, list[list[int]]]) -> None:
    profile_names = [name for name in groups_by_profile.keys() if name != "merged"]

    print("Duplicate groups by profile:")
    for profile_name in profile_names:
        groups = groups_by_profile[profile_name]
        if not groups:
            print(f"- {profile_name}: no duplicate groups")
            continue

        print(f"- {profile_name}: {len(groups)} duplicate groups")
        for idx, group in enumerate(groups, start=1):
            print(f"  {profile_name} | group_{idx}: {group}")

    merged_groups = groups_by_profile.get("merged", [])
    print("Merged duplicate-safe household groups (used for splitting):")
    for idx, group in enumerate(merged_groups, start=1):
        print(f"  merged_group_{idx}: {group}")


def main() -> None:
    _, groups_by_profile = _build_duplicate_groups(_collect_all_profiles())
    _print_duplicate_report(groups_by_profile)

    training, testing = _split_groups(groups_by_profile["merged"], target_train_size=150)
    split = {
        "training": training,
        "testing": testing,
    }
    # print(json.dumps(split, indent=4))


if __name__ == "__main__":
    main()