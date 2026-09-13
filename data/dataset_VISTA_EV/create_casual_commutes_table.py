from pathlib import Path

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


vista_dir = Path(__file__).parent
df = pd.read_csv(vista_dir / "T_VISTA1218_V1.csv")

keep_columns = ['TRIPID', 'PERSID', 'HHID', 'STOPS', 'TRIPNO', 'STARTHOUR', 'STARTIME',
       'ARRHOUR', 'ARRTIME', 'TRAVTIME', 'TRIPTIME', 'WAITIME', 'CUMDIST',
       'ORIGPLACE2', 'ORIGPURP2', 'DESTPLACE2', 'DESTPURP1', 'TRIPPURP', 'LINKMODE',
       'DIST_GRP', 'Time_Grp', 'TIME1']

# keep only columns
df_filtered = df[keep_columns]

df_casual = df_filtered[~df_filtered['TRIPPURP'].isin(['Work Related', 'Education', 'Pick-up or Drop-off Someone'])
                        &  (df_filtered['LINKMODE'] == 'Vehicle Driver')
                        & (df_filtered['ARRTIME'] < 22*60)]

# Keep only uninterrupted chains starting with TRIPNO == 1 & ORIGPLACE2 == 'Survey Home' and ending with DESTPLACE2 == 'Survey Home'
df_casual_chains = df_casual.groupby(['HHID', 'PERSID']).filter(
    lambda g: (
        g['TRIPNO'].tolist() == list(range(1, len(g) + 1))
        and g['ORIGPLACE2'].iloc[0] == 'Survey Home'
        and g['DESTPURP1'].iloc[-1] == 'Go Home'
    )
)

# save as casual_commutes.csv
df_casual_chains.to_csv(vista_dir / "casual_commutes.csv", index=False)