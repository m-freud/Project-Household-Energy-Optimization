'''
we want to replace the exisitng load profiles in energy.db
with new load profiles from the base_load_pool table of CH_loads.db

2 constraints:
- dont change the contractual power level -> keep max load in per-household allowed range
- minimize player ids used so we have enough untouched lpayers later for training

how do we do that?

two steps:
1. create candidates for each initial cp
2. 


'''

initial_cp_allowed_ranges = {
    3.45: (2.3, 3.45),
    6.9: (3.45, 6.9),
    10.35: (6.9, 10.35),
}

initial_cp_dist = {
    3.45: 152,
    6.9: 80,
    10.35: 18,
}

import sqlite3
from collections import Counter
from pathlib import Path

ch_db_path = Path(__file__).parents[2] / "sqlite" / "staging" / "CH_loads.db"

# step 1: candidates per initial cp as (player_id, date)
candidates = {}
with sqlite3.connect(ch_db_path) as conn:
    for cp, (low, high) in initial_cp_allowed_ranges.items():
        rows = conn.execute(
            "SELECT player_id, date FROM base_load_pool WHERE max_load > ? AND max_load <= ?",
            (low, high),
        ).fetchall()
        candidates[cp] = [(player_id, date) for player_id, date in rows]

# number of candidate profiles per player_id for each cp
profiles_per_player = {
    cp: dict(Counter(player_id for player_id, _ in cands).most_common())
    for cp, cands in candidates.items()
}

for cp, cands in candidates.items():
    print(f"cp {cp}: {len(cands)} candidates from {len(profiles_per_player[cp])} players")
    print(f"cp {cp}: profiles per player: {profiles_per_player[cp]}")


# step 2: select selection per initial cp with diversity awareness ( we want to use 40 player ids)

import random

rng = random.Random(42)
n_players = 40
n_players_with_7 = 10  # 30 players x 6 + 10 players x 7 = 250 profiles

dates_by_player = {
    cp: {} for cp in candidates
}
for cp, cands in candidates.items():
    for player_id, date in cands:
        dates_by_player[cp].setdefault(player_id, []).append(date)

selection = {cp: [] for cp in candidates}
cp_high, cp_mid, cp_low = 10.35, 6.9, 3.45

# highest cp: one random profile from each of the most frequent players
top_players = list(profiles_per_player[cp_high])[: initial_cp_dist[cp_high]]
for player_id in top_players:
    date = rng.choice(dates_by_player[cp_high][player_id])
    selection[cp_high].append((player_id, date))

# fill up to 40 players with those having the most candidates in the lower cps
low_cp_counts = Counter()
for cp in (cp_mid, cp_low):
    low_cp_counts.update({p: len(d) for p, d in dates_by_player[cp].items()})
other_players = [p for p, _ in low_cp_counts.most_common() if p not in top_players]
players = top_players + other_players[: n_players - len(top_players)]

# players with the most candidates overall get 7 profiles, the rest 6
total_counts = Counter()
for cp in candidates:
    total_counts.update({p: len(d) for p, d in dates_by_player[cp].items()})
players_with_7 = set(sorted(players, key=lambda p: -total_counts[p])[:n_players_with_7])
remaining = {p: 7 if p in players_with_7 else 6 for p in players}
for player_id, _ in selection[cp_high]:
    remaining[player_id] -= 1

# mid cp: round-robin over players so profiles are spread evenly
pools = {p: list(dates_by_player[cp_mid].get(p, [])) for p in players}
for pool in pools.values():
    rng.shuffle(pool)
while len(selection[cp_mid]) < initial_cp_dist[cp_mid]:
    progressed = False
    for player_id in sorted(players, key=lambda p: -remaining[p]):
        if len(selection[cp_mid]) == initial_cp_dist[cp_mid]:
            break
        if remaining[player_id] > 0 and pools[player_id]:
            selection[cp_mid].append((player_id, pools[player_id].pop()))
            remaining[player_id] -= 1
            progressed = True
    if not progressed:
        raise RuntimeError("not enough candidates for the mid cp")

# low cp: remaining quota of each player
low_pools = {p: list(dates_by_player[cp_low].get(p, [])) for p in players}
for pool in low_pools.values():
    rng.shuffle(pool)
deficit = 0
for player_id in players:
    take = min(remaining[player_id], len(low_pools[player_id]))
    deficit += remaining[player_id] - take
    for _ in range(take):
        selection[cp_low].append((player_id, low_pools[player_id].pop()))

# players lacking low-cp candidates: hand their shortfall to others, one at a time
while deficit > 0:
    spare = [p for p in players if low_pools[p]]
    if not spare:
        raise RuntimeError("not enough low-cp candidates")
    for player_id in sorted(spare, key=lambda p: (sum(q == p for q, _ in selection[cp_low]), -len(low_pools[p]))):
        if deficit == 0:
            break
        selection[cp_low].append((player_id, low_pools[player_id].pop()))
        deficit -= 1

for cp, sel in selection.items():
    print(f"cp {cp}: selected {len(sel)} of {initial_cp_dist[cp]}")
print(f"players used: {len({p for sel in selection.values() for p, _ in sel})}")

rows = [(cp, player_id, date) for cp, sel in selection.items() for player_id, date in sel]
with sqlite3.connect(ch_db_path) as conn:
    conn.execute("DROP TABLE IF EXISTS runtime_pool_by_cp")
    conn.execute(
        "CREATE TABLE runtime_pool_by_cp (init_cp REAL, player_id TEXT, date TEXT, "
        "PRIMARY KEY (player_id, date))"
    )
    conn.executemany("INSERT INTO runtime_pool_by_cp VALUES (?, ?, ?)", rows)
print(f"saved {len(rows)} rows to runtime_pool_by_cp")

print(selection)