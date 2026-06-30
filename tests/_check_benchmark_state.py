import sqlite3

conn = sqlite3.connect('sqlite/energy.db')
cur = conn.cursor()
rows = cur.execute("SELECT policy, scenario, COUNT(*) FROM results GROUP BY policy, scenario ORDER BY policy, scenario").fetchall()
print('combos=', len(rows))
print('total_rows=', sum(r[2] for r in rows))
print('has_mpc=', any(r[0] == 'mpc_oracle' for r in rows))
print('waterfall_rows=', sum(r[2] for r in rows if r[0] == 'waterfall'))
print('max_run_id=', cur.execute("SELECT MAX(CAST(run_id AS INTEGER)) FROM results").fetchone()[0])
print('tail:')
for row in rows[-8:]:
    print(row)
conn.close()
