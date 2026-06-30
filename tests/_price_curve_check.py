import math
import pandas as pd
from src.sqlite_connection import load_series

PLAYER_ID = 1
SCENARIO = 'default_scenario'
POLICIES = ['waterfall', 'mpc_oracle']

price = load_series('buy_price', PLAYER_ID, SCENARIO)
if price.empty:
    raise SystemExit('No buy_price series found')

print(f'Price curve check for household {PLAYER_ID}, scenario {SCENARIO}')
print(f"{'policy':<12} {'avg_buy':>10} {'avg_load':>10} {'load_wtd_price':>15} {'corr(load,price)':>17} {'low_price_load%':>16}")
print('-' * 88)

for policy in POLICIES:
    load = load_series('net_load', PLAYER_ID, SCENARIO, policy)
    if load.empty:
        print(f'{policy:<12} no data')
        continue

    df = pd.DataFrame({'hour': load['hour'], 'load': load['value'], 'price': price['value']})
    df['abs_load'] = df['load'].abs()
    total_abs_load = df['abs_load'].sum()
    load_wtd_price = (df['price'] * df['abs_load']).sum() / total_abs_load if total_abs_load else math.nan
    corr = df['load'].corr(df['price'])
    avg_buy = df['price'].mean()
    avg_load = df['load'].mean()

    q25 = df['price'].quantile(0.25)
    low_price_mask = df['price'] <= q25
    low_price_load = df.loc[low_price_mask, 'abs_load'].sum() / total_abs_load * 100 if total_abs_load else math.nan

    print(f'{policy:<12} {avg_buy:>10.4f} {avg_load:>10.4f} {load_wtd_price:>15.4f} {corr:>17.4f} {low_price_load:>16.1f}%')

print('\nTop 8 lowest-price periods with load')
for policy in POLICIES:
    load = load_series('net_load', PLAYER_ID, SCENARIO, policy)
    df = pd.DataFrame({'period': load['period'], 'hour': load['hour'], 'load': load['value'], 'price': price['value']})
    df = df.sort_values('price').head(8)
    print(f'\n{policy}:')
    for _, row in df.iterrows():
        print(f"  period={int(row['period']):>2} hour={row['hour']:>4.2f} price={row['price']:>7.4f} load={row['load']:>8.4f}")
