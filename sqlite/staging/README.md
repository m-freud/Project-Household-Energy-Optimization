# Staging Databases Overview

This folder contains the transformed, clean, and normalized SQLite staging databases for each target domain before scenario selection and assembly.

| Database | Tables | Format / Details | Dimensions / Volume |
| :--- | :--- | :--- | :--- |
| **`CH_loads.db`** | `load` | Wide format (`timestamp_utc`, `period`, `<player_id>` columns) in kW | ~55 MB (Swiss smart meter data across time) |
| **`DE_pv_gen.db`** | `pv_gen` | Wide format (`timestamp_utc`, `period`, `residential1..6` columns) in kW | ~5 MB (German OPSD residential PV profiles) |
| **`VISTA_ev1_status.db`** | `ev1_status`, `ev1_distances` | Wide format (622 commute pairs with `period` 1..96 and person columns) | ~0.2 MB (0 = home, 1 = driving, 2 = work) |
| **`VISTA_ev2_status.db`** | `ev2_status`, `ev2_distances` | Long format (`persid`, `period`, `status`) and (`persid`, `trip_no`, `distance`) | ~37 MB (7,789 persons, 747k rows) |

---

### Note on EV2 Format:
`ev2_status` and `ev2_distances` use a **long format** (`persid`, `period`, `status`) instead of the wide format used in `ev1_status`. 

**Reason:** With 7,789 unique persons in the casual commutes dataset, a wide table with one column per person would exceed SQLite's maximum column limit (2,000 columns). The long format is indexed on `(persid, period)` for fast lookups during scenario assembly.


### Note for selection / training:

Make sure to exclude days that include NaN values or very high peaks (cap at 10kW or sth like that)