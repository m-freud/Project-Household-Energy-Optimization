import sqlite3
from pathlib import Path
import random

ROOT_DIR = Path(__file__).resolve().parents[2]
STAGING_DIR = ROOT_DIR / "sqlite" / "staging"
CH_LOADS_DB = STAGING_DIR / "CH_loads.db"
DE_PV_DB = STAGING_DIR / "DE_pv_gen.db"
EV1_DB = STAGING_DIR / "VISTA_ev1_status.db"
EV2_DB = STAGING_DIR / "VISTA_ev2_status.db"
ENERGY_V2_DB = ROOT_DIR / "sqlite" / "energy_v2.db"
TRAINING_DB = ROOT_DIR / "sqlite" / "energy_v2_training.db"

PV_EXCLUDED_PLAYER = "residential3"
PV_BEFORE_YEAR = 2018
PV_FACTOR = 0.8


def create_load_table(conn: sqlite3.Connection) -> None:
    """load: only CH households that are not in runtime_pool_by_cp."""
    conn.execute("ATTACH DATABASE ? AS ch", (str(CH_LOADS_DB),))
    try:
        columns = [r[1] for r in conn.execute("PRAGMA ch.table_info(load)")][2:]
        used = {r[0] for r in conn.execute("SELECT DISTINCT player_id FROM ch.runtime_pool_by_cp")}
        training = [c for c in columns if c not in used]
        select_cols = ", ".join(f'"{c}"' for c in ["timestamp_utc", "period", *training])

        conn.execute('DROP TABLE IF EXISTS main."load"')
        conn.execute(f'CREATE TABLE main."load" AS SELECT {select_cols} FROM ch."load" ORDER BY timestamp_utc')
        print(f"load: {len(training)} households")
    finally:
        conn.commit()
        conn.execute("DETACH DATABASE ch")


def create_pv_gen_table(conn: sqlite3.Connection) -> None:
    """pv_gen: all DE profiles before PV_BEFORE_YEAR except PV_EXCLUDED_PLAYER, scaled by PV_FACTOR."""
    conn.execute("ATTACH DATABASE ? AS de", (str(DE_PV_DB),))
    try:
        columns = [r[1] for r in conn.execute("PRAGMA de.table_info(pv_gen)")][2:]
        training = [c for c in columns if c != PV_EXCLUDED_PLAYER]
        scaled = ", ".join(f'"{c}" * {PV_FACTOR} AS "{c}"' for c in training)

        conn.execute('DROP TABLE IF EXISTS main."pv_gen"')
        conn.execute(
            f'CREATE TABLE main."pv_gen" AS SELECT timestamp_utc, period, {scaled} '
            "FROM de.pv_gen WHERE timestamp_utc < ? ORDER BY timestamp_utc",
            (f"{PV_BEFORE_YEAR}-01-01",),
        )
        print(f"pv_gen: {training}")
    finally:
        conn.commit()
        conn.execute("DETACH DATABASE de")


def create_e1_status_table(conn: sqlite3.Connection) -> None:
    """ev1_status: all VISTA profiles from households not used in runtime."""
    conn.execute("ATTACH DATABASE ? AS vista", (str(EV1_DB),))
    try:
        profile_ids = [row[1] for row in conn.execute('PRAGMA vista.table_info("ev1_status")')][1:]
        persons = dict(conn.execute("SELECT persid, HHID FROM vista.ev1_persons"))
        shared_ids = {
            persid
            for (persid,) in conn.execute(
                "SELECT persid FROM vista.ev1_persons "
                "WHERE HHID IN ("
                "SELECT HHID FROM vista.ev1_persons GROUP BY HHID "
                "HAVING COUNT(DISTINCT persid) > 1)"
            )
        }
        preferred = [pid for pid in profile_ids if persons[pid] and pid in shared_ids]
        remaining = [pid for pid in profile_ids if pid not in shared_ids]
        rng = random.Random(42)
        rng.shuffle(preferred)
        rng.shuffle(remaining)
        runtime_ids = (preferred + remaining)[:250]
        runtime_hhids = {persons[pid] for pid in runtime_ids}
        training_ids = [pid for pid in profile_ids if persons[pid] not in runtime_hhids]

        selected_columns = ", ".join(f'"{profile_id}"' for profile_id in training_ids)
        conn.execute('DROP TABLE IF EXISTS main."ev1_status"')
        conn.execute(
            f'CREATE TABLE main."ev1_status" AS '
            f'SELECT period, {selected_columns} FROM vista."ev1_status" ORDER BY period'
        )
        print(f"ev1_status: {len(training_ids)} training profiles, "
              f"excluding {len(runtime_hhids)} runtime HHIDs")
    finally:
        conn.commit()
        conn.execute("DETACH DATABASE vista")


def create_e2_status_table(conn: sqlite3.Connection) -> None:
    """ev2_status: 1,000 random profiles from HHIDs not used by runtime."""
    conn.execute("ATTACH DATABASE ? AS vista", (str(EV2_DB),))
    conn.execute("ATTACH DATABASE ? AS runtime", (str(ENERGY_V2_DB),))
    try:
        candidates = conn.execute(
            "SELECT persid, HHID FROM vista.ev2_persons ORDER BY persid"
        ).fetchall()
        runtime_hhids = {
            row[0] for row in conn.execute("SELECT HHID FROM runtime.ev2_persons")
        }

        rng = random.Random(42)
        rng.shuffle(candidates)
        eligible = []
        used_hhids = set(runtime_hhids)
        for persid, hhid in candidates:
            if hhid not in used_hhids:
                eligible.append((persid, hhid))
                used_hhids.add(hhid)
        if len(eligible) < 1000:
            raise RuntimeError(f"Found only {len(eligible)} eligible EV2 HHIDs, need 1000")
        selected = eligible[:1000]

        columns = ", ".join(f'"{i}" INTEGER' for i in range(1, 1001))
        conn.execute('DROP TABLE IF EXISTS main."ev2_status"')
        conn.execute(
            f'CREATE TABLE main."ev2_status" ("period" INTEGER PRIMARY KEY, {columns})'
        )
        conn.executemany(
            'INSERT INTO main."ev2_status" (period) VALUES (?)',
            [(period,) for period in range(1, 97)],
        )

        conn.execute('DROP TABLE IF EXISTS main."ev2_persons"')
        conn.execute(
            "CREATE TABLE main.ev2_persons "
            "(player_id INTEGER PRIMARY KEY, persid TEXT, HHID TEXT UNIQUE)"
        )
        for player_id, (persid, hhid) in enumerate(selected, start=1):
            rows = conn.execute(
                "SELECT period, status FROM vista.ev2_status WHERE persid = ? ORDER BY period",
                (persid,),
            ).fetchall()
            if len(rows) != 96:
                raise ValueError(f"EV2 profile {persid}: expected 96 periods, got {len(rows)}")
            conn.executemany(
                f'UPDATE main."ev2_status" SET "{player_id}" = ? WHERE period = ?',
                [(status, period) for period, status in rows],
            )
            conn.execute(
                "INSERT INTO main.ev2_persons (player_id, persid, HHID) VALUES (?, ?, ?)",
                (player_id, persid, hhid),
            )

        conn.commit()
        print(f"ev2_status: {len(selected)} training profiles from unique non-runtime HHIDs")
    finally:
        conn.execute("DETACH DATABASE vista")
        conn.execute("DETACH DATABASE runtime")


def main():
    TRAINING_DB.unlink(missing_ok=True)
    conn = sqlite3.connect(TRAINING_DB)
    try:
        create_load_table(conn)
        create_pv_gen_table(conn)
        create_e1_status_table(conn)
        create_e2_status_table(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
