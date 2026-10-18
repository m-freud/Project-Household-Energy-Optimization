# Runtime / Training Data Split

**Status:** Beschreibung des aktuellen Ingestion-Codes und der aktuell erzeugten SQLite-Datenbanken (Stand 2026-10-18).

## Overview

The project prepares four profile targets for runtime and training:

1. Household base load (`base_load` / `load`)
2. PV generation (`pv_gen`)
3. EV1 work-commute status (`ev1_status`)
4. EV2 casual-commute status (`ev2_status`)

Runtime data is written to `sqlite/energy_v2.db`. Training data is written to `sqlite/energy_v2_training.db`. The scripts preserve each target's established simulation-facing representation: the four runtime targets are wide, with one column per player and 96 period rows. Training load and PV remain timestamped wide time series; EV1 and EV2 training statuses are also stored wide.

A profile count needs interpretation by target. Load runtime has 250 target players but draws from 40 distinct source households. PV runtime has 200 active PV targets but cycles through a 92-day source pool. EV1 is selected by person, with shared households intentionally prioritized. EV2 requires one person per household.

## Database Build Sequence

### Runtime

`src/ingestion/insert_external_profiles.py` runs these steps in order:

1. `copy_energy_db()` replaces `energy_v2.db` with a SQLite backup of `energy.db`.
2. It drops every table whose name ends in `_metadata`. In the current source DB these are `base_load_metadata`, `pv_gen_metadata`, `ev1_metadata`, and `ev2_metadata`.
3. It replaces the selected source data for `base_load`, `pv_gen`, `ev1_status`, and `ev2_status`.

The PV source pool must be built first by running `data/dataset_DE_PV/select_ext_runtime_profiles.py`. This creates `runtime_pool` in `sqlite/staging/DE_pv_gen.db`.

### Training

`src/ingestion/create_training_tables.py` deletes and recreates `energy_v2_training.db`, then builds `load`, `pv_gen`, `ev1_status`, `ev2_status`, and the EV2 `ev2_persons` mapping. It does not copy the full runtime energy database or its metadata/configuration tables.

## 1. Household Base Load

### Source data

`sqlite/staging/CH_loads.db`, table `load`:

- Wide table with `timestamp_utc`, `period`, and 81 numeric household-ID columns.
- 70,176 rows, equivalent to 731 days x 96 periods.
- Current timestamps span `2022-12-31 23:00:00` through `2024-12-31 22:45:00`.

The same database contains `runtime_pool_by_cp` with 250 `(init_cp, player_id, date)` rows, drawn from 40 distinct CH household IDs. The rows by CP are:

| Initial CP | Runtime days | Distinct source households |
|---:|---:|---:|
| 3.45 | 152 | 37 |
| 6.90 | 80 | 39 |
| 10.35 | 18 | 18 |
| **Total / union** | **250** | **40** |

Households can occur under multiple CP groups, so the distinct counts by group do not sum to the union.

### Runtime selection and transform

For each target player ID 1-250, the insertion script:

1. Reads that player's `initial_cp` from `energy.db`.
2. Selects the next unused `(player_id, date)` row from the corresponding `runtime_pool_by_cp` group. Within each CP group the rows are sorted by numeric source household ID and then date.
3. Extracts the source household's 96 values for that date, ordered by `period`.
4. Writes those values into the target player's column in `energy_v2.db.base_load`.

The source day is used once. Multiple days from one CH household can supply different runtime players. Runtime has **250 target profiles** and **40 distinct source households represented**. Each profile is one 96-period day.

### Training selection

Training keeps all CH household columns whose IDs do not appear in `runtime_pool_by_cp`. This excludes the 40 runtime source households as whole household series, not just the selected days.

- **41 training household profiles**
- 70,176 timestamps per column, spanning 731 x 96 periods
- Result table: `energy_v2_training.db.load`

## 2. PV Generation

### Source data

`sqlite/staging/DE_pv_gen.db`, table `pv_gen`:

- Wide table with `timestamp_utc`, `period`, and four source series: `residential1`, `residential3`, `residential4`, `residential6`.
- 117,393 rows in the source table.

`pv_gen_pool` contains per-player, per-day candidate metadata. It has 156 rows for `residential1`, 268 for `residential3`, 184 for `residential4`, and 212 for `residential6` in the current staging database.

### Runtime selection and transform

First run `data/dataset_DE_PV/select_ext_runtime_profiles.py`. Its `create_runtime_pool()` takes all `pv_gen_pool` rows for `residential3` in 2018, ordered chronologically:

- **92 daily profiles**, from 2018-03-16 through 2018-09-19.

`replace_pv_gen()` reads the 200 player IDs in `energy.db.pv_gen_metadata`, sorted by player ID. It assigns the runtime pool sequentially and cycles to the beginning after exhausting the 92 days, so 200 targets are filled from 92 distinct source days. For each target, it reads the selected `residential3` day from `pv_gen` and multiplies every value by **0.8**. The target's original PV metadata is not used for scaling in the current code.

- **200 active PV target profiles**
- **92 distinct source day profiles**, reused cyclically
- The wide runtime table still has 250 player columns; 200 are the PV targets selected from metadata.

### Training selection and transform

`create_pv_gen_table()` selects every timestamp before `2018-01-01` from raw `pv_gen`, and retains the three columns `residential1`, `residential4`, and `residential6`. It excludes the entire `residential3` series and multiplies the retained values by **0.8**.

Current output:

- **3 source-series profiles**
- 91,714 rows, 956 distinct dates, from 2015-05-21 through 2017-12-31.
- Each series has 91,714 non-null values: 955 complete 96-period days plus one partial day (the first date starts at 15:30).
- This selection is by source player and date, not by household ID.

## 3. EV1 Work-Commute Status

### Source data

`data/dataset_VISTA_EV/work_commute_pairs.csv` is transformed by `create_ev1_database.py` into `sqlite/staging/VISTA_ev1_status.db`.

The processed source contains:

- **622 person profiles** (`PERSID`)
- **603 distinct household IDs** (`HHID`)
- 19 HHIDs shared by two persons each, i.e. **38 people** in shared households
- `ev1_status`: 96 period rows and one column per PERSID (wide)
- `ev1_distances`: long trip-level data with `persid`, `HHID`, `trip_no`, `distance`
- `ev1_persons`: one `persid` / `HHID` mapping row per person

The daily status values are 0 (home), 1 (driving), or 2 (parked away / at work).

### Runtime selection

`replace_ev1_status()` uses a fixed `random.Random(42)` selection to make the assignment reproducible:

1. It places all 38 people belonging to shared HHIDs in the preferred list.
2. It separately shuffles that list and all other people with the seeded RNG.
3. It selects the preferred people first, then fills target player IDs 1-250 from the shuffled remainder.
4. It copies each 96-value status profile into the matching wide player column.

Current split:

- **250 runtime person profiles**
- **231 runtime HHIDs**; the 19 shared HHIDs are deliberately represented by both people
- The runtime EV1 `ev1_status` table is 96 rows x 251 columns (period plus players 1-250).

The runtime EV1 `HHID` assignment is not currently persisted in `energy_v2.db`; it is reconstructed from the source mapping and the fixed selection rule when creating the training table.

### Training selection

`create_e1_status_table()` reproduces the runtime selection order and determines the selected runtime HHIDs. It then includes every source person whose HHID is not in that runtime-HHID set. This excludes households as a group, even when a household has multiple people.

- **372 training person profiles**
- No training HHID overlaps runtime
- `ev1_status`: 96 rows x 373 columns (period plus 372 PERSID columns)
- Training EV1 does not currently include a separate `ev1_persons` mapping table; HHID is used during selection but the output remains the same wide status format as the source/runtime.

## 4. EV2 Casual-Commute Status

### Source data

`data/dataset_VISTA_EV/casual_commutes.csv` is transformed by `create_ev2_database.py` into `sqlite/staging/VISTA_ev2_status.db`.

The processed source contains:

- **7,789 person profiles**
- **7,054 distinct HHIDs**
- 747,744 status rows = 7,789 x 96 periods
- `ev2_status`: long format (`persid`, `HHID`, `period`, `status`)
- `ev2_distances`: long trip-level data (`persid`, `HHID`, `trip_no`, `distance`)
- `ev2_persons`: one `persid` / `HHID` mapping per person

As for EV1, status values are 0 (home), 1 (driving), and 2 (parked away / at destination).

### Runtime selection

`replace_ev2_status()` sorts candidates by PERSID, shuffles them with `random.Random(42)`, and then takes the first profile from each previously unused HHID until 250 profiles are selected. It copies the 96 statuses into player columns 1-250 in `energy_v2.ev2_status`.

- **250 runtime profiles**
- **250 distinct runtime HHIDs**
- Runtime `ev2_status`: wide, 96 rows x 251 columns (period plus players 1-250)
- Runtime `ev2_persons`: 250 mapping rows (`player_id`, `persid`, `HHID`), with a unique-HHID constraint

### Training selection

`create_e2_status_table()` starts from the full source person list, shuffles with seed 42, excludes every HHID in runtime `ev2_persons`, then takes the first 1,000 remaining unique HHIDs.

- **1,000 training profiles**
- **1,000 distinct training HHIDs**, disjoint from runtime
- 6,804 distinct source HHIDs remain eligible after excluding the 250 runtime households; 1,000 are sampled for training
- Training `ev2_status`: wide, 96 rows x 1,001 columns (period plus players 1-1000)
- Training `ev2_persons`: 1,000 mapping rows (`player_id`, `persid`, `HHID`), with a unique-HHID constraint

Training EV2 is a sample of eligible households, not the full remainder of the EV2 source pool.

## Summary Counts

| Target | Source profiles | Runtime profiles | Runtime distinct source HHIDs / inputs | Training profiles | Training HHID policy |
|---|---:|---:|---:|---:|---|
| Base load | 81 CH households | 250 target days | 40 CH households | 41 household series | Excludes the 40 runtime CH household IDs |
| PV generation | 4 DE series; runtime pool has 92 days | 200 target profiles | 92 source days, cyclic assignment | 3 source series x 955 complete days (plus one partial day each) | All pre-2018 data except residential3; scaled 0.8 |
| EV1 status | 622 people / 603 HHIDs | 250 people | 231 HHIDs; 38 preferred shared-HHID people are first | 372 people | Excludes every runtime HHID |
| EV2 status | 7,789 people / 7,054 HHIDs | 250 people | 250 HHIDs | 1,000 people | 1,000 distinct non-runtime HHIDs; sampled from 6,804 eligible |

## Final Table Formats

`wide` means one profile per data column. `long` means profile identity is repeated in rows with period/value fields.

| Target / database | Format | Current shape / key columns |
|---|---|---|
| Staging CH `load` | Wide | 70,176 rows; `timestamp_utc`, `period`, 81 household columns |
| Runtime `base_load` | Wide | 96 rows; `period` + 250 player columns |
| Training `load` | Wide | 70,176 rows; `timestamp_utc`, `period` + 41 household columns |
| Staging DE `pv_gen` | Wide | 117,393 rows; `timestamp_utc`, `period` + 4 source-series columns |
| Runtime `pv_gen` | Wide | 96 rows; `period` + 250 player columns (200 PV targets) |
| Training `pv_gen` | Wide | 91,714 rows; `timestamp_utc`, `period` + 3 PV source-series columns |
| Staging EV1 `ev1_status` | Wide | 96 rows; `period` + 622 PERSID columns |
| Runtime EV1 `ev1_status` | Wide | 96 rows; `period` + 250 player columns |
| Training EV1 `ev1_status` | Wide | 96 rows; `period` + 372 PERSID columns |
| Staging EV2 `ev2_status` | Long | 747,744 rows; `persid`, `HHID`, `period`, `status` |
| Runtime EV2 `ev2_status` | Wide | 96 rows; `period` + 250 player columns |
| Training EV2 `ev2_status` | Wide | 96 rows; `period` + 1,000 player columns |
| EV1/EV2 distances | Long | Person/household ID plus trip number and distance |
| EV2 runtime/training `ev2_persons` | Long mapping | One row per assigned player: `player_id`, `persid`, `HHID` |

## Reproduction Commands

Run from the repository root in the project's virtual environment:

```powershell
# Rebuild EV1 and EV2 source staging databases, including HHIDs
.\.venv\Scripts\python.exe data\dataset_VISTA_EV\create_ev1_database.py
.\.venv\Scripts\python.exe data\dataset_VISTA_EV\create_ev2_database.py

# Rebuild the 92-day PV runtime pool before runtime insertion
.\.venv\Scripts\python.exe data\dataset_DE_PV\select_ext_runtime_profiles.py

# Build runtime database (copies energy.db, drops metadata, replaces all four targets)
.\.venv\Scripts\python.exe src\ingestion\insert_external_profiles.py

# Build training database
.\.venv\Scripts\python.exe src\ingestion\create_training_tables.py
```

## Important Reproducibility Notes

- Rebuild the VISTA staging databases before runtime/training if their source CSVs change; these scripts generate the HHID mapping tables used for EV selection.
- Rebuild the PV runtime pool before running runtime insertion. The runtime insertion script consumes the existing `runtime_pool` and does not create it.
- EV1 training reconstructs runtime household membership from the source plus the same fixed seed-42 selection code. Unlike EV2, EV1 runtime currently does not persist a player-to-PERSID/HHID mapping in `energy_v2.db`.
- Runtime EV1 intentionally prioritizes shared-HHID people, so it is not one-person-per-household. Runtime and training HHID sets remain disjoint.
- PV uses a fixed factor of 0.8 in both runtime and training. Runtime does not currently scale PV profiles to original `total_energy`; the earlier total-energy scaling expression is inactive.
- `energy_v2_training.db` is rebuilt from scratch. Its current targets are load, PV, EV1, and EV2; it is not a full copy of `energy_v2.db`.
