import sqlite3
db = sqlite3.connect("sqlite/energy.db")
rows = db.execute(
    "SELECT policy, scenario, COUNT(*) as n, MIN(player_id), MAX(player_id) "
    "FROM results WHERE run_id='4' "
    "GROUP BY policy, scenario ORDER BY policy, scenario"
).fetchall()
print(f"run_id=4: {len(rows)} combos")
for r in rows:
    print(f"  {r[0]:<22} {r[1]:<30} n={r[2]}  hh={r[3]}..{r[4]}")
db.close()
