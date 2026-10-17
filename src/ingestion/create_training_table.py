import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
STAGING_DIR = ROOT_DIR / "sqlite" / "staging"
CH_LOADS_DB = STAGING_DIR / "CH_loads.db"
DE_PV_DB = STAGING_DIR / "DE_pv_gen.db"
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


def main():
    TRAINING_DB.unlink(missing_ok=True)
    conn = sqlite3.connect(TRAINING_DB)
    try:
        create_load_table(conn)
        create_pv_gen_table(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
