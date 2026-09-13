# print FULL content of head(20) of T_VISTA1218_V1.csv

import pandas as pd
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

vista_dir = Path(__file__).parent
df = pd.read_csv(vista_dir / "T_VISTA1218_V1.csv")

keep_columns = ['TRIPID', 'PERSID', 'HHID', 'STOPS', 'TRIPNO', 'STARTHOUR', 'STARTIME',
       'ARRHOUR', 'ARRTIME', 'TRAVTIME', 'TRIPTIME', 'WAITIME',
       'ORIGPLACE2', 'ORIGPURP2', 'DESTPLACE2','DESTPURP2', 'TRIPPURP', 'LINKMODE',
       'DIST_GRP', 'Time_Grp', 'TIME1']

# keep only columns
df_filtered = df[keep_columns]

df_work_related = df_filtered[(df_filtered['TRIPPURP'] == 'Work Related') & (df_filtered['LINKMODE'] == 'Vehicle Driver')]

EV_COMMUTE_WINDOWS_ALLOWED = {
    "ev1": {
        "window1": {"earliest_start": 32, "latest_end": 50, "max_unavailable_steps": 5},
        "window2": {"earliest_start": 70, "latest_end": 88, "max_unavailable_steps": 5},
    },
}

df_allowed_work_commutes = df_work_related[
    (
    (df_work_related['STARTIME'] >= EV_COMMUTE_WINDOWS_ALLOWED['ev1']['window1']['earliest_start'] * 15) &
    (df_work_related['ARRTIME'] <= EV_COMMUTE_WINDOWS_ALLOWED['ev1']['window1']['latest_end'] * 15) &
    (df_work_related['ORIGPLACE2'] == 'Survey Home') &
    (df_work_related['DESTPLACE2'] == 'My Workplace') &
    (df_work_related['TRIPNO'] == 1)
    ) | (
    (df_work_related['STARTIME'] >= EV_COMMUTE_WINDOWS_ALLOWED['ev1']['window2']['earliest_start'] * 15) &
    (df_work_related['ARRTIME'] <= EV_COMMUTE_WINDOWS_ALLOWED['ev1']['window2']['latest_end'] * 15) &
    (df_work_related['ORIGPLACE2'] == 'My Workplace') &
    (df_work_related['DESTPLACE2'] == 'Survey Home') &
    (df_work_related['TRIPNO'] == 2)
    )
]

df_allowed_pairs = df_allowed_work_commutes.groupby(['HHID', 'PERSID']).filter(lambda x: len(x) == 2)

# save as work_commute_pairs.csv
df_allowed_pairs.to_csv( vista_dir / "work_commute_pairs.csv", index=False)