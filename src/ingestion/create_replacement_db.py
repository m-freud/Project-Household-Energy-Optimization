
import sqlite3
from collections import defaultdict
import math
from pathlib import Path
import random

ROOT_DIR = Path(__file__).resolve().parents[2]
SQLITE_DIR = ROOT_DIR / "sqlite"
ENERGY_DB = SQLITE_DIR / "energy.db"
ENERGY_V2_DB = SQLITE_DIR / "energy_v2.db"
CH_LOADS_DB = SQLITE_DIR / "staging" / "CH_loads.db"
DE_PV_DB = SQLITE_DIR / "staging" / "DE_pv_gen.db"
EV1_DB = SQLITE_DIR / "staging" / "VISTA_ev1_status.db"
EV2_DB = SQLITE_DIR / "staging" / "VISTA_ev2_status.db"

TIME_SERIES_TABLES_TO_REPLACE = (
    "base_load",
    "pv_gen",
    "ev1_buy_price",
    "ev2_buy_price",
    "ev1_load",
    "ev2_load",
    "ev1_max_charge",
    "ev2_max_charge",
    "ev1_at_home",
    "ev2_at_home",
    "ev1_at_charging_station",
    "ev2_at_charging_station",
)

N_PLAYERS = 250
N_PERIODS = 96
PERIOD_HOURS = 0.25


def copy_energy_db() -> None:
    """Copy energy.db without metadata or time series that will be replaced."""
    ENERGY_V2_DB.unlink(missing_ok=True)
    src = sqlite3.connect(ENERGY_DB)
    dst = sqlite3.connect(ENERGY_V2_DB)
    try:
        src.backup(dst)
        tables = dst.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE '%\\_metadata' ESCAPE '\\'"
        ).fetchall()
        for (name,) in tables:
            dst.execute(f'DROP TABLE "{name}"')
        for table_name in TIME_SERIES_TABLES_TO_REPLACE:
            dst.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        dst.commit()
        dst.execute("VACUUM")
    finally:
        src.close()
        dst.close()


def load_pool_by_cp() -> dict[float, list[tuple[str, str]]]:
    """cp -> [(player_id, date)] sorted by player_id."""
    conn = sqlite3.connect(CH_LOADS_DB)
    try:
        rows = conn.execute("SELECT init_cp, player_id, date FROM runtime_pool_by_cp").fetchall()
    finally:
        conn.close()
    pool: dict[float, list[tuple[str, str]]] = defaultdict(list)
    for cp, player_id, date in rows:
        pool[float(cp)].append((player_id, date))
    for entries in pool.values():
        entries.sort(key=lambda e: (int(e[0]), e[1]))
    return dict(pool)


def read_initial_cps(conn: sqlite3.Connection) -> dict[int, float]:
    rows = conn.execute("SELECT player_id, initial_cp FROM initial_cp").fetchall()
    return {int(pid): float(cp) for pid, cp in rows}


def read_ch_day(ch_conn: sqlite3.Connection, ch_player_id: str, date: str) -> list[float]:
    # CH player ids are numeric strings used as column names
    if not ch_player_id.isdigit():
        raise ValueError(f"Invalid CH player id: {ch_player_id}")
    rows = ch_conn.execute(
        f'SELECT "{ch_player_id}" FROM load WHERE substr(timestamp_utc, 1, 10) = ? ORDER BY period',
        (date,),
    ).fetchall()
    if len(rows) != N_PERIODS:
        raise ValueError(f"{ch_player_id} {date}: expected {N_PERIODS} rows, got {len(rows)}")
    return [r[0] for r in rows]


def ensure_base_load_table(conn: sqlite3.Connection) -> None:
    columns = ",\n  ".join(f'"{i}" REAL' for i in range(1, N_PLAYERS + 1))
    conn.execute(f'CREATE TABLE IF NOT EXISTS "base_load" (\n"period" INTEGER,\n  {columns}\n)')
    if conn.execute('SELECT COUNT(*) FROM "base_load"').fetchone()[0] == 0:
        conn.executemany(
            'INSERT INTO "base_load" (period) VALUES (?)',
            [(p,) for p in range(1, N_PERIODS + 1)],
        )


def replace_base_load() -> None:
    pool = load_pool_by_cp()
    next_idx: dict[float, int] = defaultdict(int)

    conn = sqlite3.connect(ENERGY_V2_DB)
    ch_conn = sqlite3.connect(CH_LOADS_DB)
    try:
        initial_cps = read_initial_cps(conn)
        ensure_base_load_table(conn)

        for player_id in range(1, N_PLAYERS + 1):
            cp = initial_cps[player_id]
            idx = next_idx[cp]
            if idx >= len(pool.get(cp, [])):
                raise RuntimeError(f"CP pool {cp} exhausted at player {player_id}")
            next_idx[cp] += 1
            ch_player_id, date = pool[cp][idx]

            values = read_ch_day(ch_conn, ch_player_id, date)
            conn.executemany(
                f'UPDATE "base_load" SET "{player_id}" = ? WHERE period = ?',
                [(v, p) for p, v in enumerate(values, start=1)],
            )
            print(f"player {player_id:3d} (cp {cp}) <- CH {ch_player_id} {date}")
        conn.commit()
    finally:
        conn.close()
        ch_conn.close()


def replace_pv_gen() -> None:
    """Create pv_gen and populate PV players from the external runtime pool."""
    conn = sqlite3.connect(ENERGY_V2_DB)
    de_conn = sqlite3.connect(DE_PV_DB)
    orig_conn = sqlite3.connect(ENERGY_DB)
    try:
        targets = orig_conn.execute(
            "SELECT player_id, total_energy FROM pv_gen_metadata ORDER BY player_id"
        ).fetchall()
        pool = de_conn.execute(
            "SELECT player_id, date, total_energy FROM runtime_pool ORDER BY date"
        ).fetchall()
        if not targets:
            raise RuntimeError("pv_gen_metadata is empty")
        if not pool:
            raise RuntimeError("runtime_pool is empty")

        target_ids = {int(player_id) for player_id, _ in targets}
        invalid_target_ids = target_ids - set(range(1, N_PLAYERS + 1))
        if invalid_target_ids:
            raise ValueError(f"Invalid PV player IDs: {sorted(invalid_target_ids)}")

        columns = ",\n  ".join(
            f'"{player_id}" REAL' for player_id in range(1, N_PLAYERS + 1)
        )
        conn.execute('DROP TABLE IF EXISTS "pv_gen"')
        conn.execute(
            f'CREATE TABLE "pv_gen" ("period" INTEGER PRIMARY KEY,\n  {columns})'
        )
        player_columns = ", ".join(
            f'"{player_id}"' for player_id in range(1, N_PLAYERS + 1)
        )
        placeholders = ", ".join("?" for _ in range(N_PLAYERS + 1))
        conn.executemany(
            f'INSERT INTO "pv_gen" ("period", {player_columns}) VALUES ({placeholders})',
            [
                (period, *([0.0] * N_PLAYERS))
                for period in range(1, N_PERIODS + 1)
            ],
        )

        # cycle through the pool again once all days are used
        for i, (player_id, total_orig) in enumerate(targets):
            src_id, date, total_new = pool[i % len(pool)]
            if not src_id.replace("_", "").isalnum():
                raise ValueError(f"Invalid DE player id: {src_id}")
            rows = de_conn.execute(
                f'SELECT "{src_id}" FROM pv_gen WHERE substr(timestamp_utc, 1, 10) = ? ORDER BY period',
                (date,),
            ).fetchall()
            if len(rows) != N_PERIODS:
                raise ValueError(f"{src_id} {date}: expected {N_PERIODS} rows, got {len(rows)}")

            factor = 0.8
            conn.executemany(
                f'UPDATE "pv_gen" SET "{int(player_id)}" = ? WHERE period = ?',
                [(r[0] * factor, p) for p, r in enumerate(rows, start=1)],
            )
            print(f"pv player {player_id:3d} <- DE {src_id} {date} (x{factor:.3f})")
        conn.commit()
    finally:
        conn.close()
        de_conn.close()
        orig_conn.close()


def replace_ev1_status() -> None:
    """Fill energy_v2 EV1 players, prioritizing people from shared households."""
    conn = sqlite3.connect(ENERGY_V2_DB)
    ev1_conn = sqlite3.connect(EV1_DB)
    try:
        selected = _select_ev1_runtime_profiles(ev1_conn)

        for player_id, (source_id, hhid) in enumerate(selected, start=1):
            rows = ev1_conn.execute(
                f'SELECT "{source_id}" FROM ev1_status ORDER BY period'
            ).fetchall()
            if len(rows) != N_PERIODS:
                raise ValueError(f"EV1 profile {source_id}: expected {N_PERIODS} rows, got {len(rows)}")
            conn.executemany(
                f'UPDATE "ev1_status" SET "{player_id}" = ? WHERE period = ?',
                [(row[0], period) for period, row in enumerate(rows, start=1)],
            )
            print(f"EV1 player {player_id:3d} <- {source_id} (HHID {hhid})")
        conn.commit()
    finally:
        conn.close()
        ev1_conn.close()


def _select_ev1_runtime_profiles(
    ev1_conn: sqlite3.Connection,
) -> list[tuple[str, str]]:
    profile_ids = [
        row[1] for row in ev1_conn.execute('PRAGMA table_info("ev1_status")')
    ][1:]
    persons = dict(ev1_conn.execute("SELECT persid, HHID FROM ev1_persons"))
    shared_ids = {
        persid
        for (persid,) in ev1_conn.execute(
            "SELECT persid FROM ev1_persons "
            "WHERE HHID IN ("
            "SELECT HHID FROM ev1_persons GROUP BY HHID "
            "HAVING COUNT(DISTINCT persid) > 1)"
        )
    }

    preferred = [pid for pid in profile_ids if persons.get(pid) and pid in shared_ids]
    remaining = [pid for pid in profile_ids if pid not in shared_ids]
    rng = random.Random(42)
    rng.shuffle(preferred)
    rng.shuffle(remaining)
    selected_ids = (preferred + remaining)[:N_PLAYERS]
    if len(selected_ids) != N_PLAYERS:
        raise RuntimeError(
            f"Found {len(selected_ids)} EV1 profiles, need {N_PLAYERS}"
        )
    return [(persid, persons[persid]) for persid in selected_ids]


def replace_ev2_status() -> None:
    """Fill runtime EV2 players with random profiles from distinct households."""
    conn = sqlite3.connect(ENERGY_V2_DB)
    ev2_conn = sqlite3.connect(EV2_DB)
    try:
        candidates = ev2_conn.execute(
            "SELECT persid, HHID FROM ev2_persons ORDER BY persid"
        ).fetchall()
        random.Random(42).shuffle(candidates)
        selected = []
        used_hhids = set()
        for persid, hhid in candidates:
            if hhid not in used_hhids:
                selected.append((persid, hhid))
                used_hhids.add(hhid)
            if len(selected) == N_PLAYERS:
                break
        if len(selected) != N_PLAYERS:
            raise RuntimeError(f"Found only {len(selected)} unique EV2 households, need {N_PLAYERS}")

        conn.execute("DROP TABLE IF EXISTS ev2_persons")
        conn.execute(
            "CREATE TABLE ev2_persons (player_id INTEGER PRIMARY KEY, persid TEXT, HHID TEXT UNIQUE)"
        )
        for player_id, (persid, hhid) in enumerate(selected, start=1):
            rows = ev2_conn.execute(
                "SELECT period, status FROM ev2_status WHERE persid = ? ORDER BY period",
                (persid,),
            ).fetchall()
            if len(rows) != N_PERIODS:
                raise ValueError(f"EV2 profile {persid}: expected {N_PERIODS} periods, got {len(rows)}")
            conn.executemany(
                f'UPDATE "ev2_status" SET "{player_id}" = ? WHERE period = ?',
                [(status, period) for period, status in rows],
            )
            conn.execute(
                "INSERT INTO ev2_persons (player_id, persid, HHID) VALUES (?, ?, ?)",
                (player_id, persid, hhid),
            )
            print(f"EV2 player {player_id:3d} <- {persid} (HHID {hhid})")
        conn.commit()
    finally:
        conn.close()
        ev2_conn.close()


def _replace_ev_availability(
    status_table: str,
    at_home_table: str,
    at_station_table: str,
    station_status: int | None,
) -> None:
    conn = sqlite3.connect(ENERGY_V2_DB)
    try:
        columns = [
            row[1]
            for row in conn.execute(f'PRAGMA table_info("{status_table}")')
        ]
        if not columns or columns[0] != "period":
            raise ValueError(f"{status_table} must have period as its first column")

        profile_columns = columns[1:]
        expected_profile_columns = [str(player_id) for player_id in range(1, N_PLAYERS + 1)]
        if profile_columns != expected_profile_columns:
            raise ValueError(
                f"{status_table}: expected household columns 1..{N_PLAYERS}"
            )

        quoted_profiles = ", ".join(f'"{column}"' for column in profile_columns)
        rows = conn.execute(
            f'SELECT "period", {quoted_profiles} FROM "{status_table}" ORDER BY "period"'
        ).fetchall()
        if len(rows) != N_PERIODS or [row[0] for row in rows] != list(
            range(1, N_PERIODS + 1)
        ):
            raise ValueError(
                f"{status_table}: expected periods 1..{N_PERIODS}, "
                f"found {len(rows)} rows"
            )

        valid_statuses = {0, 1, 2}
        for row in rows:
            invalid_values = {
                value for value in row[1:] if value is None or value not in valid_statuses
            }
            if invalid_values:
                raise ValueError(
                    f"{status_table} contains invalid status values: "
                    f"{sorted(invalid_values, key=str)}"
                )

        def create_availability_table(table_name: str, values: list[tuple]) -> None:
            conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            profile_definitions = ", ".join(
                f'"{column}" INTEGER' for column in profile_columns
            )
            conn.execute(
                f'CREATE TABLE "{table_name}" '
                f'("period" INTEGER PRIMARY KEY, {profile_definitions})'
            )
            conn.executemany(
                f'INSERT INTO "{table_name}" '
                f'("period", {quoted_profiles}) '
                f'VALUES ({", ".join("?" for _ in range(N_PLAYERS + 1))})',
                values,
            )

        home_rows = [
            (row[0], *(int(status == 0) for status in row[1:]))
            for row in rows
        ]
        if station_status is None:
            station_rows = [
                (row[0], *([0] * N_PLAYERS))
                for row in rows
            ]
        else:
            station_rows = [
                (row[0], *(int(status == station_status) for status in row[1:]))
                for row in rows
            ]

        conn.execute("BEGIN")
        create_availability_table(at_home_table, home_rows)
        create_availability_table(at_station_table, station_rows)
        conn.commit()
        print(f"Created {at_home_table} and {at_station_table} from {status_table}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def replace_ev1_availability() -> None:
    """Derive EV1 home and charging-station profiles from EV1 status."""
    _replace_ev_availability(
        status_table="ev1_status",
        at_home_table="ev1_at_home",
        at_station_table="ev1_at_charging_station",
        station_status=2,
    )


def replace_ev2_availability() -> None:
    """Derive EV2 home profiles; EV2 has no charging-station availability."""
    _replace_ev_availability(
        status_table="ev2_status",
        at_home_table="ev2_at_home",
        at_station_table="ev2_at_charging_station",
        station_status=None,
    )


def _replace_ev_load(
    ev_num: int,
    source_db_path: Path,
    assignments: list[tuple[int, str]],
    wide_status: bool,
) -> None:
    conn = sqlite3.connect(ENERGY_V2_DB)
    source_conn = sqlite3.connect(source_db_path)
    try:
        status_table = f"ev{ev_num}_status"
        ev_table = f"ev{ev_num}"
        distance_table = f"ev{ev_num}_distances"
        status_columns = [
            row[1] for row in conn.execute(f'PRAGMA table_info("{status_table}")')
        ]
        expected_columns = ["period"] + [
            str(player_id) for player_id in range(1, N_PLAYERS + 1)
        ]
        if status_columns != expected_columns:
            raise ValueError(
                f"{status_table}: expected household columns 1..{N_PLAYERS}"
            )
        if len(assignments) != N_PLAYERS or [
            player_id for player_id, _ in assignments
        ] != list(range(1, N_PLAYERS + 1)):
            raise ValueError(f"EV{ev_num} profile mapping must contain players 1..{N_PLAYERS}")

        rate_rows = conn.execute(
            f'SELECT player_id, consumption_per_km FROM "{ev_table}" ORDER BY player_id'
        ).fetchall()
        consumption_by_player = {
            int(player_id): float(consumption_per_km)
            for player_id, consumption_per_km in rate_rows
        }
        if set(consumption_by_player) != set(range(1, N_PLAYERS + 1)):
            raise ValueError(f"{ev_table} must contain players 1..{N_PLAYERS}")

        profile_loads: dict[int, list[float]] = {}
        status_player_columns = ", ".join(
            f'"{player_id}"' for player_id in range(1, N_PLAYERS + 1)
        )
        runtime_status_rows = conn.execute(
            f'SELECT "period", {status_player_columns} '
            f'FROM "{status_table}" ORDER BY "period"'
        ).fetchall()
        if len(runtime_status_rows) != N_PERIODS or [
            row[0] for row in runtime_status_rows
        ] != list(range(1, N_PERIODS + 1)):
            raise ValueError(f"{status_table}: expected periods 1..{N_PERIODS}")
        runtime_status_by_player = {
            player_id: [row[player_id] for row in runtime_status_rows]
            for player_id in range(1, N_PLAYERS + 1)
        }

        for player_id, persid in assignments:
            if wide_status:
                if not persid.replace("_", "").isalnum():
                    raise ValueError(f"Invalid EV1 profile id: {persid}")
                source_rows = source_conn.execute(
                    f'SELECT "{persid}" FROM "{status_table}" ORDER BY "period"'
                ).fetchall()
            else:
                source_rows = source_conn.execute(
                    f'SELECT "status" FROM "{status_table}" '
                    'WHERE "persid" = ? ORDER BY "period"',
                    (persid,),
                ).fetchall()

            source_status = [row[0] for row in source_rows]
            if len(source_status) != N_PERIODS:
                raise ValueError(
                    f"EV{ev_num} profile {persid}: expected {N_PERIODS} status rows, "
                    f"got {len(source_status)}"
                )
            if source_status != runtime_status_by_player[player_id]:
                raise ValueError(
                    f"EV{ev_num} status profile for player {player_id} does not match "
                    f"VISTA profile {persid}"
                )
            if any(status not in (0, 1, 2) for status in source_status):
                raise ValueError(f"EV{ev_num} profile {persid} has invalid status values")

            distance_rows = source_conn.execute(
                f'SELECT "distance" FROM "{distance_table}" '
                'WHERE "persid" = ? ORDER BY "trip_no"',
                (persid,),
            ).fetchall()
            if any(distance is None or not math.isfinite(float(distance)) or distance < 0
                   for (distance,) in distance_rows):
                raise ValueError(f"EV{ev_num} profile {persid} has invalid distances")
            if any(status == 1 for status in source_status) and not distance_rows:
                raise ValueError(
                    f"EV{ev_num} profile {persid} has driving periods but no distances"
                )

            driving_periods = sum(status == 1 for status in source_status)
            total_distance_km = sum(float(row[0]) for row in distance_rows)
            consumption_per_km = consumption_by_player[player_id]
            if not math.isfinite(consumption_per_km) or consumption_per_km < 0:
                raise ValueError(
                    f"EV{ev_num} player {player_id} has invalid consumption_per_km"
                )
            power_kw = (
                total_distance_km * consumption_per_km / 1000
                / (driving_periods * PERIOD_HOURS)
                if driving_periods
                else 0.0
            )
            profile_loads[player_id] = [
                power_kw if status == 1 else 0.0 for status in source_status
            ]

        profile_definitions = ", ".join(
            f'"{player_id}" REAL' for player_id in range(1, N_PLAYERS + 1)
        )
        conn.execute("BEGIN")
        conn.execute(f'DROP TABLE IF EXISTS "ev{ev_num}_load"')
        conn.execute(
            f'CREATE TABLE "ev{ev_num}_load" '
            f'("period" INTEGER PRIMARY KEY, {profile_definitions})'
        )
        placeholders = ", ".join("?" for _ in range(N_PLAYERS + 1))
        conn.executemany(
            f'INSERT INTO "ev{ev_num}_load" VALUES ({placeholders})',
            [
                (
                    period,
                    *(profile_loads[player_id][period - 1]
                      for player_id in range(1, N_PLAYERS + 1)),
                )
                for period in range(1, N_PERIODS + 1)
            ],
        )
        conn.commit()
        print(f"Created ev{ev_num}_load from VISTA distances and EV consumption rates")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
        source_conn.close()


def replace_ev1_load() -> None:
    """Create EV1 driving load evenly across each selected profile's driving periods."""
    ev1_conn = sqlite3.connect(EV1_DB)
    try:
        selected = _select_ev1_runtime_profiles(ev1_conn)
    finally:
        ev1_conn.close()
    assignments = [
        (player_id, persid)
        for player_id, (persid, _) in enumerate(selected, start=1)
    ]
    _replace_ev_load(1, EV1_DB, assignments, wide_status=True)


def replace_ev2_load() -> None:
    """Create EV2 driving load evenly across each selected profile's driving periods."""
    with sqlite3.connect(ENERGY_V2_DB) as conn:
        assignments = [
            (int(player_id), str(persid))
            for player_id, persid in conn.execute(
                'SELECT "player_id", "persid" FROM "ev2_persons" ORDER BY "player_id"'
            )
        ]
    _replace_ev_load(2, EV2_DB, assignments, wide_status=False)


def _replace_ev_max_charge(ev_num: int, has_station: bool) -> None:
    conn = sqlite3.connect(ENERGY_V2_DB)
    try:
        expected_profile_columns = [
            str(player_id) for player_id in range(1, N_PLAYERS + 1)
        ]
        home_columns = [
            row[1] for row in conn.execute('PRAGMA table_info("ev{}_at_home")'.format(ev_num))
        ]
        if home_columns != ["period", *expected_profile_columns]:
            raise ValueError(f"ev{ev_num}_at_home must contain periods and players 1..{N_PLAYERS}")

        station_table = f"ev{ev_num}_at_charging_station"
        station_columns = [
            row[1] for row in conn.execute(f'PRAGMA table_info("{station_table}")')
        ]
        if station_columns != ["period", *expected_profile_columns]:
            raise ValueError(
                f"{station_table} must contain periods and players 1..{N_PLAYERS}"
            )

        home_rows = conn.execute(
            f'SELECT * FROM "ev{ev_num}_at_home" ORDER BY "period"'
        ).fetchall()
        station_rows = conn.execute(
            f'SELECT * FROM "{station_table}" ORDER BY "period"'
        ).fetchall()
        expected_periods = list(range(1, N_PERIODS + 1))
        if (
            len(home_rows) != N_PERIODS
            or [row[0] for row in home_rows] != expected_periods
            or len(station_rows) != N_PERIODS
            or [row[0] for row in station_rows] != expected_periods
        ):
            raise ValueError(
                f"EV{ev_num} availability tables must contain periods 1..{N_PERIODS}"
            )

        home_rate_rows = conn.execute(
            'SELECT "player_id", "max_home_charge" '
            'FROM "max_home_charge" ORDER BY "player_id"'
        ).fetchall()
        home_rates = {
            int(player_id): float(rate)
            for player_id, rate in home_rate_rows
            if rate is not None
        }
        if set(home_rates) != set(range(1, N_PLAYERS + 1)):
            raise ValueError(
                f"max_home_charge must contain non-null values for players 1..{N_PLAYERS}"
            )

        station_rates = {player_id: 0.0 for player_id in range(1, N_PLAYERS + 1)}
        if has_station:
            station_rows_by_player = conn.execute(
                f'SELECT "player_id", "station_max_charge" '
                f'FROM "ev{ev_num}" ORDER BY "player_id"'
            ).fetchall()
            station_rates = {
                int(player_id): float(rate)
                for player_id, rate in station_rows_by_player
                if rate is not None
            }
            if set(station_rates) != set(range(1, N_PLAYERS + 1)):
                raise ValueError(
                    f"ev{ev_num}.station_max_charge must contain non-null values "
                    f"for players 1..{N_PLAYERS}"
                )

        for player_id in range(1, N_PLAYERS + 1):
            for field, rates in (
                ("home", home_rates),
                ("station", station_rates),
            ):
                rate = rates[player_id]
                if not math.isfinite(rate) or rate < 0:
                    raise ValueError(
                        f"EV{ev_num} player {player_id} has invalid {field} charge rate"
                    )

        result_rows = []
        for home_row, station_row in zip(home_rows, station_rows):
            values = []
            for index, player_id in enumerate(range(1, N_PLAYERS + 1), start=1):
                at_home = home_row[index]
                at_station = station_row[index]
                if at_home not in (0, 1) or at_station not in (0, 1):
                    raise ValueError(
                        f"EV{ev_num} availability values must be 0 or 1"
                    )
                if at_home and at_station:
                    raise ValueError(
                        f"EV{ev_num} player {player_id} cannot be home and at a station "
                        f"in period {home_row[0]}"
                    )
                values.append(
                    home_rates[player_id] if at_home
                    else station_rates[player_id] if at_station
                    else 0.0
                )
            result_rows.append((home_row[0], *values))

        profile_definitions = ", ".join(
            f'"{player_id}" REAL' for player_id in range(1, N_PLAYERS + 1)
        )
        table_name = f"ev{ev_num}_max_charge"
        placeholders = ", ".join("?" for _ in range(N_PLAYERS + 1))
        conn.execute("BEGIN")
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        conn.execute(
            f'CREATE TABLE "{table_name}" '
            f'("period" INTEGER PRIMARY KEY, {profile_definitions})'
        )
        conn.executemany(
            f'INSERT INTO "{table_name}" VALUES ({placeholders})',
            result_rows,
        )
        conn.commit()
        print(f"Created {table_name} from home and station charging limits")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def replace_ev1_max_charge() -> None:
    """Use home charging limits and EV1 public-station rates by location."""
    _replace_ev_max_charge(1, has_station=True)


def replace_ev2_max_charge() -> None:
    """Use EV2 home charging limits; EV2 has no station charging."""
    _replace_ev_max_charge(2, has_station=False)



def main():
    print("Ingesting load profiles")
    # copy_energy_db()
    # replace_base_load()
    # replace_pv_gen()
    # replace_ev1_status()
    # replace_ev2_status()
    replace_ev1_availability() # backwards comp
    replace_ev2_availability() # backwards comp
    replace_ev1_load()
    replace_ev2_load()
    replace_ev1_max_charge()
    replace_ev2_max_charge()




if __name__ == "__main__":
    main()