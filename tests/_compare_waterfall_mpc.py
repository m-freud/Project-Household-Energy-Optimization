import sqlite3

scenarios = [
    'default_scenario',
    'early_urgency',
    'high_start_narrow',
    'late_relaxed',
    'low_start_wide',
    'mid_start_normal',
    'stressed_ev_buffered_bess',
]
metrics = ['total_cost', 'total_consumption', 'net_cost', 'net_load']

conn = sqlite3.connect('sqlite/energy.db')
cur = conn.cursor()

print(f"{'scenario':<28} {'metric':<18} {'waterfall':>12} {'mpc':>12} {'delta':>12} {'delta%':>8}")
print('-' * 92)
for scen in scenarios:
    for metric in metrics:
        waterfall = cur.execute(
            f"SELECT AVG({metric}) FROM results WHERE policy='waterfall' AND scenario=?",
            (scen,),
        ).fetchone()[0]
        mpc = cur.execute(
            f"SELECT AVG({metric}) FROM results WHERE policy='mpc_oracle' AND scenario=?",
            (scen,),
        ).fetchone()[0]
        delta = mpc - waterfall
        pct = (delta / waterfall * 100) if waterfall else 0.0
        print(f"{scen:<28} {metric:<18} {waterfall:>12.4f} {mpc:>12.4f} {delta:>+12.4f} {pct:>7.2f}%")
    print('-' * 92)

conn.close()
