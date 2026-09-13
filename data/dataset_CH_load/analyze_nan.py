'''
Analyze NaN gaps in smart_meter_data_filtered csvs (column: kWh_to_installation).

For each household csv, reports:
- total NaN count
- number of NaN chains (consecutive-NaN runs) longer than 2, 4, 6, 8 samples
- number of distinct days containing a NaN chain longer than 2, 4, 6, 8 samples

Prints a per-household table plus an overall summary.
'''

from pathlib import Path

import pandas as pd

DATASET_DIR = Path(__file__).parent
SMART_METER_DATA_DIR = DATASET_DIR / "smart_meter_data_filtered"
VALUE_COLUMN = "kWh_to_installation"
CHAIN_THRESHOLDS = (2, 4, 6, 8)


def analyze_file(path: Path) -> dict:
    df = pd.read_csv(path, sep=";", parse_dates=["timestamp_utc"])
    values = df[VALUE_COLUMN]
    is_nan = values.isna()

    total_nan = int(is_nan.sum())

    # assign a unique id to each consecutive run, then keep only NaN runs
    run_id = (is_nan != is_nan.shift()).cumsum()
    nan_runs = df.loc[is_nan, ["timestamp_utc"]].assign(run_id=run_id[is_nan])
    run_lengths = nan_runs.groupby("run_id").size()
    run_days = nan_runs.groupby("run_id")["timestamp_utc"].agg(lambda ts: ts.dt.date.iloc[0])

    chains_over = {}
    days_over = {}
    for threshold in CHAIN_THRESHOLDS:
        long_run_ids = run_lengths[run_lengths > threshold].index
        chains_over[threshold] = int(len(long_run_ids))
        days_over[threshold] = int(run_days.loc[long_run_ids].nunique())

    return {
        "household_id": path.stem,
        "rows": len(df),
        "total_nan": total_nan,
        "chains_over": chains_over,
        "days_over": days_over,
    }


def main():
    files = sorted(SMART_METER_DATA_DIR.glob("*.csv"))
    results = [analyze_file(f) for f in files]

    header = (
        f"{'household_id':>12} | {'rows':>7} | {'nan':>6} | "
        + " | ".join(f"chains>{t:>2}" for t in CHAIN_THRESHOLDS)
        + " | "
        + " | ".join(f"days>{t:>2}  " for t in CHAIN_THRESHOLDS)
    )
    print(header)
    print("-" * len(header))
    for r in results:
        chains_str = " | ".join(f"{r['chains_over'][t]:>8}" for t in CHAIN_THRESHOLDS)
        days_str = " | ".join(f"{r['days_over'][t]:>7}" for t in CHAIN_THRESHOLDS)
        print(f"{r['household_id']:>12} | {r['rows']:>7} | {r['total_nan']:>6} | {chains_str} | {days_str}")

    print("\n=== Summary across all households ===")
    print(f"households analyzed: {len(results)}")
    print(f"total nan values: {sum(r['total_nan'] for r in results)}")
    for t in CHAIN_THRESHOLDS:
        total_chains = sum(r["chains_over"][t] for r in results)
        total_days = sum(r["days_over"][t] for r in results)
        print(f"  chains longer than {t}: {total_chains} total, across {total_days} household-days")


if __name__ == "__main__":
    main()
'''
household_id |    rows |    nan | chains> 2 | chains> 4 | chains> 6 | chains> 8 | days> 2   | days> 4   | days> 6   | days> 8  
-------------------------------------------------------------------------------------------------------------------------------
      100354 |   70176 |     11 |        1 |        0 |        0 |        0 |       1 |       0 |       0 |       0
      100707 |   70176 |     10 |        0 |        0 |        0 |        0 |       0 |       0 |       0 |       0
      102016 |   70176 |     40 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      104481 |   70176 |     35 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      105366 |   70176 |     26 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      106298 |   70176 |     30 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      106793 |   70176 |     40 |        2 |        1 |        1 |        1 |       2 |       1 |       1 |       1
      107448 |   70176 |     24 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      108517 |   70176 |     23 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      108795 |   70176 |     39 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      110260 |   70176 |     96 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      112377 |   70176 |     96 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      113377 |   70176 |      6 |        0 |        0 |        0 |        0 |       0 |       0 |       0 |       0
      115937 |   70176 |     96 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      116320 |   70176 |     96 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      118984 |   70176 |     27 |        2 |        1 |        1 |        1 |       2 |       1 |       1 |       1
      120517 |   70176 |     96 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      121500 |   70176 |      0 |        0 |        0 |        0 |        0 |       0 |       0 |       0 |       0
      121555 |   70176 |     22 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      123129 |   70176 |    149 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      125463 |   70176 |    126 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      127225 |   70176 |     34 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      127524 |   70176 |     36 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      128843 |   70176 |    103 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      129071 |   70176 |    151 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      130253 |   70176 |    172 |        2 |        2 |        2 |        2 |       2 |       2 |       2 |       2
      132256 |   70176 |      8 |        0 |        0 |        0 |        0 |       0 |       0 |       0 |       0
      132967 |   70176 |      8 |        0 |        0 |        0 |        0 |       0 |       0 |       0 |       0
      133165 |   70176 |     33 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      133996 |   70176 |      9 |        1 |        0 |        0 |        0 |       1 |       0 |       0 |       0
      136407 |   70176 |     11 |        1 |        0 |        0 |        0 |       1 |       0 |       0 |       0
      138414 |   70176 |     49 |        2 |        2 |        2 |        2 |       2 |       2 |       2 |       2
      139410 |   70176 |     26 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      140211 |   70176 |     24 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      140649 |   70176 |    127 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      142975 |   70176 |     33 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      142999 |   70176 |      8 |        0 |        0 |        0 |        0 |       0 |       0 |       0 |       0
      144492 |   70176 |     10 |        0 |        0 |        0 |        0 |       0 |       0 |       0 |       0
      146624 |   70176 |      0 |        0 |        0 |        0 |        0 |       0 |       0 |       0 |       0
      147049 |   70176 |      0 |        0 |        0 |        0 |        0 |       0 |       0 |       0 |       0
      147147 |   70176 |     22 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      151683 |   70176 |     41 |        2 |        1 |        1 |        1 |       2 |       1 |       1 |       1
      151740 |   70176 |      9 |        1 |        0 |        0 |        0 |       1 |       0 |       0 |       0
      152601 |   70176 |     62 |        2 |        2 |        2 |        2 |       2 |       2 |       2 |       2
      155209 |   70176 |     34 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      157904 |   70176 |      8 |        0 |        0 |        0 |        0 |       0 |       0 |       0 |       0
      160216 |   70176 |     26 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      160431 |   70176 |     10 |        2 |        0 |        0 |        0 |       2 |       0 |       0 |       0
      161573 |   70176 |      4 |        0 |        0 |        0 |        0 |       0 |       0 |       0 |       0
      163329 |   70176 |     96 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      163595 |   70176 |    151 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      163817 |   70176 |      0 |        0 |        0 |        0 |        0 |       0 |       0 |       0 |       0
      164897 |   70176 |    129 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      165786 |   70176 |     96 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      166084 |   70176 |    125 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      166206 |   70176 |     32 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      167494 |   70176 |     33 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      168165 |   70176 |      9 |        1 |        0 |        0 |        0 |       1 |       0 |       0 |       0
      169310 |   70176 |     23 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      170013 |   70176 |      9 |        1 |        0 |        0 |        0 |       1 |       0 |       0 |       0
      171530 |   70176 |      0 |        0 |        0 |        0 |        0 |       0 |       0 |       0 |       0
      172262 |   70176 |     34 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      172510 |   70176 |    172 |        2 |        2 |        2 |        2 |       2 |       2 |       2 |       2
      174968 |   70176 |     96 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      176350 |   70176 |    149 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      179723 |   70176 |     24 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      180146 |   70176 |     23 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      180931 |   70176 |    129 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      183144 |   70176 |     22 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      184973 |   70176 |    172 |        2 |        2 |        2 |        2 |       2 |       2 |       2 |       2
      185196 |   70176 |    127 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      187422 |   70176 |    151 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      188093 |   70176 |     21 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      188131 |   70176 |     97 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      188208 |   70176 |      0 |        0 |        0 |        0 |        0 |       0 |       0 |       0 |       0
      189416 |   70176 |     96 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      191628 |   70176 |     38 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      191701 |   70176 |     26 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      192815 |   70176 |     96 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      198650 |   70176 |     96 |        1 |        1 |        1 |        1 |       1 |       1 |       1 |       1
      199888 |   70176 |     71 |        3 |        3 |        2 |        2 |       3 |       3 |       2 |       2

=== Summary across all households ===
households analyzed: 81
total nan values: 4489
  chains longer than 2: 78 total, across 78 household-days
  chains longer than 4: 67 total, across 67 household-days
  chains longer than 6: 66 total, across 66 household-days
  chains longer than 8: 66 total, across 66 household-days

'''