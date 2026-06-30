import math
import pandas as pd
from src.sqlite_connection import load_series, load_household_ids

SCENARIO = 'default_scenario'
POLICIES = ['waterfall', 'mpc_oracle']
HOUSEHOLDS = [1, 2, 3, 4, 5]

print(f'Battery wear check for scenario {SCENARIO}')
print(f"{'hh':>3} {'policy':<12} {'bess_abs_throughput':>20} {'bess_net':>10} {'bess_max_abs':>13} {'bess_cycles_eq':>15}")
print('-' * 78)

for hh in HOUSEHOLDS:
    for policy in POLICIES:
        bess = load_series('bess_power', hh, SCENARIO, policy)
        if bess.empty:
            continue
        abs_throughput = bess['value'].abs().sum() * 0.25
        net = bess['value'].sum() * 0.25
        max_abs = bess['value'].abs().max()
        # rough equivalent full cycles proxy: throughput / (2 * capacity)
        # capacity is not directly in series, so we print a normalized proxy based on kWh throughput only
        print(f'{hh:>3} {policy:<12} {abs_throughput:>20.4f} {net:>10.4f} {max_abs:>13.4f} {abs_throughput:>15.4f}')

print('\nAggregate across sampled households')
for policy in POLICIES:
    thr = []
    mx = []
    for hh in HOUSEHOLDS:
        bess = load_series('bess_power', hh, SCENARIO, policy)
        if bess.empty:
            continue
        thr.append(bess['value'].abs().sum() * 0.25)
        mx.append(bess['value'].abs().max())
    if thr:
        print(f'{policy:<12} avg_abs_throughput={sum(thr)/len(thr):.4f}  avg_peak_abs={sum(mx)/len(mx):.4f}')
