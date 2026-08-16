# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))


from src.sqlite_connection import load_attribute


def _has_bess(player_id: int) -> bool:
    return bool(load_attribute("player_pv_bess", player_id, "has_bess"))


def _has_pv(player_id: int) -> bool:
    return bool(load_attribute("player_pv_bess", player_id, "has_pv"))


def print_ids_with_pv_bess(ids: list[int]) -> None:
    player_with_bess = []
    player_with_pv = []

    for player_id in sorted(ids):
        if _has_bess(player_id):
            player_with_bess.append(player_id)
        if _has_pv(player_id):
            player_with_pv.append(player_id)

    print(f"Players with BESS: {player_with_bess}")
    print(f"Players with PV: {player_with_pv}")

print("=== Players with PV and/or BESS in the runtime test folds ===")
print_ids_with_pv_bess(list(range(1, 251)))