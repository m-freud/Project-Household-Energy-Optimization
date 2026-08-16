# find equal groups for all 4 prediciton metrics
# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))


from src.sqlite_connection import fetch_timeseries, sqlite_cursor

def _normalize_series(series: list[float]) -> list[float]:
    min_val = min(series)
    max_val = max(series)
    if max_val - min_val == 0:
        return [0.0 for _ in series]
    return [(value - min_val) / (max_val - min_val) for value in series]


def _equivalent_series(s1: list[float], s2: list[float], tolerance: float = 0.00001) -> bool:
    if len(s1) != len(s2):
        return False
    for idx in range(len(s1)):
        if abs(float(s1[idx]) - float(s2[idx])) > tolerance:
            return False
    return True


def _series_profile(player_id: int, table_name: str) -> list[float]:
    values = fetch_timeseries(sqlite_cursor, player_id=player_id, table_name=table_name)
    return _normalize_series(values)



def find_equal_groups(table_name:str):
    profiles = {}
    for id in range(1, 251):
        profiles[id] = _series_profile(id, table_name)

    assigned_ids = []
    equality_groups = {}
    for i in range(1, 251):
        for j in range(1, 251):
            if i == j:
                continue

            if j in assigned_ids:
                continue

            s1 = profiles[i]
            s2 = profiles[j]

            if _equivalent_series(s1, s2):
                assigned_ids.append(i)
                assigned_ids.append(j)
                if i not in equality_groups.keys():
                    equality_groups[i] = [i, j]
                else:
                    equality_groups[i].append(j)

    independent_ids = [id for id in range(1, 251) if id not in assigned_ids]

    for id in independent_ids:
        equality_groups[id] = [id]

    return equality_groups


eg_base_load = find_equal_groups("base_load")
eg_pv_gen = find_equal_groups("pv_gen")
eg_ev1 = find_equal_groups("ev1_status")
eg_ev2 = find_equal_groups("ev2_status")

# print("BASE LOAD", eg_base_load)
# print("PV GEN", eg_pv_gen)
# print("EV1 STATUS", eg_ev1)
# print("EV2 STATUS", eg_ev2)


def plot_group_distributions():
    """
    categorical distributions of equality-group sizes for each prediction metric
    """
    import matplotlib.pyplot as plt

    def plot_group_sizes(equality_groups, title):
        group_keys = [str(group_key) for group_key in equality_groups.keys()]
        group_sizes = [len(group) for group in equality_groups.values()]

        plt.figure(figsize=(14, 6))
        plt.bar(group_keys, group_sizes, color="blue", edgecolor="black", alpha=0.75)
        plt.title(title)
        plt.xlabel("Group key (first household id)")
        plt.ylabel("Number of households in group")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.75)
        plt.tight_layout()
        plt.show()

    plot_group_sizes(eg_base_load, "Base Load Equality Group Distribution")
    plot_group_sizes(eg_pv_gen, "PV Generation Equality Group Distribution")
    plot_group_sizes(eg_ev1, "EV1 Status Equality Group Distribution")
    plot_group_sizes(eg_ev2, "EV2 Status Equality Group Distribution")

if __name__ == "__main__":
    plot_group_distributions()