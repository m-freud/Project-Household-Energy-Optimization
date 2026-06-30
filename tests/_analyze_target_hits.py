import sqlite3
from collections import defaultdict

conn = sqlite3.connect('sqlite/energy.db')
cur = conn.cursor()

policies = [row[0] for row in cur.execute("SELECT DISTINCT policy FROM results ORDER BY policy")]
scenarios = [row[0] for row in cur.execute("SELECT DISTINCT scenario FROM results ORDER BY scenario")]

print('=== Overall by policy ===')
print(f"{'policy':<22} {'n':>5} {'cost':>10} {'load':>10} {'bess_all':>10} {'ev1_all':>10} {'ev2_all':>10} {'bess_final':>11} {'ev1_final':>11} {'ev2_final':>11}")
for policy in policies:
    row = cur.execute(
        """
        SELECT
            COUNT(*) AS n,
            AVG(total_cost) AS cost,
            AVG(total_consumption) AS load,
            AVG(CASE WHEN target_met_all_bess THEN 1.0 ELSE 0.0 END) AS bess_all,
            AVG(CASE WHEN target_met_all_ev1 THEN 1.0 ELSE 0.0 END) AS ev1_all,
            AVG(CASE WHEN target_met_all_ev2 THEN 1.0 ELSE 0.0 END) AS ev2_all,
            AVG(CASE WHEN target_met_bess THEN 1.0 ELSE 0.0 END) AS bess_final,
            AVG(CASE WHEN target_met_ev1 THEN 1.0 ELSE 0.0 END) AS ev1_final,
            AVG(CASE WHEN target_met_ev2 THEN 1.0 ELSE 0.0 END) AS ev2_final
        FROM results
        WHERE policy = ?
        """,
        (policy,),
    ).fetchone()
    print(f"{policy:<22} {row[0]:>5} {row[1]:>10.4f} {row[2]:>10.4f} {row[3]*100:>9.1f}% {row[4]*100:>9.1f}% {row[5]*100:>9.1f}% {row[6]*100:>10.1f}% {row[7]*100:>10.1f}% {row[8]*100:>10.1f}%")

print('\n=== By policy and scenario: MPC vs waterfall ===')
for scenario in scenarios:
    print(f"\nScenario: {scenario}")
    print(f"{'policy':<22} {'n':>5} {'cost':>10} {'load':>10} {'bess_all':>10} {'ev1_all':>10} {'ev2_all':>10}")
    for policy in ['waterfall', 'mpc_oracle', 'fast_charge', 'even_linear', 'price_aware_linear', 'no_control']:
        row = cur.execute(
            """
            SELECT
                COUNT(*) AS n,
                AVG(total_cost) AS cost,
                AVG(total_consumption) AS load,
                AVG(CASE WHEN target_met_all_bess THEN 1.0 ELSE 0.0 END) AS bess_all,
                AVG(CASE WHEN target_met_all_ev1 THEN 1.0 ELSE 0.0 END) AS ev1_all,
                AVG(CASE WHEN target_met_all_ev2 THEN 1.0 ELSE 0.0 END) AS ev2_all
            FROM results
            WHERE policy = ? AND scenario = ?
            """,
            (policy, scenario),
        ).fetchone()
        if row[0] == 0:
            continue
        print(f"{policy:<22} {row[0]:>5} {row[1]:>10.4f} {row[2]:>10.4f} {row[3]*100:>9.1f}% {row[4]*100:>9.1f}% {row[5]*100:>9.1f}%")

print('\n=== MPC breakdown by has_pv / has_bess ===')
print(f"{'scenario':<28} {'pv':<5} {'bess':<5} {'n':>5} {'cost':>10} {'load':>10} {'bess_all':>10} {'ev1_all':>10} {'ev2_all':>10}")
for scenario in scenarios:
    rows = cur.execute(
        """
        SELECT has_pv, has_bess, COUNT(*) AS n,
               AVG(total_cost), AVG(total_consumption),
               AVG(CASE WHEN target_met_all_bess THEN 1.0 ELSE 0.0 END),
               AVG(CASE WHEN target_met_all_ev1 THEN 1.0 ELSE 0.0 END),
               AVG(CASE WHEN target_met_all_ev2 THEN 1.0 ELSE 0.0 END)
        FROM results
        WHERE policy = 'mpc_oracle' AND scenario = ?
        GROUP BY has_pv, has_bess
        ORDER BY has_pv, has_bess
        """,
        (scenario,),
    ).fetchall()
    for has_pv, has_bess, n, cost, load, bess_all, ev1_all, ev2_all in rows:
        print(f"{scenario:<28} {int(has_pv):<5} {int(has_bess):<5} {n:>5} {cost:>10.4f} {load:>10.4f} {bess_all*100:>9.1f}% {ev1_all*100:>9.1f}% {ev2_all*100:>9.1f}%")

conn.close()
