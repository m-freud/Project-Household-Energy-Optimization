import sqlite3

conn = sqlite3.connect('sqlite/energy.db')
cur = conn.cursor()
scenarios = [row[0] for row in cur.execute("SELECT DISTINCT scenario FROM results ORDER BY scenario")]
policies = ['waterfall', 'mpc_oracle']

print(f"{'scenario':<28} {'policy':<12} {'n':>5} {'avg_abs_bess_throughput':>24} {'avg_peak_abs_power':>20}")
print('-' * 84)
for scen in scenarios:
    for policy in policies:
        row = cur.execute(
            """
         SELECT COUNT(DISTINCT player_id) AS n,
             AVG(abs_throughput) AS avg_abs_throughput,
             AVG(peak_abs_power) AS avg_peak_abs_power
            FROM (
          SELECT player_id,
              SUM(ABS(value)) * 0.25 AS abs_throughput,
              MAX(ABS(value)) AS peak_abs_power
                FROM bess_power
                WHERE scenario = ? AND policy = ?
                GROUP BY player_id
            )
            """,
            (scen, policy),
        ).fetchone()
        if row[0] == 0:
            continue
        print(f"{scen:<28} {policy:<12} {row[0]:>5} {row[1]:>24.4f} {row[2]:>20.4f}")

conn.close()
