
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SQLITE_DIR = ROOT_DIR / "sqlite"
ENERGY_DB = SQLITE_DIR / "energy.db"
ENERGY_V2_DB = SQLITE_DIR / "energy_v2.db"
CH_LOADS_DB = SQLITE_DIR / "staging" / "CH_loads.db"
DE_PV_DB = SQLITE_DIR / "staging" / "DE_pv_gen.db"

N_PLAYERS = 250
N_PERIODS = 96


def copy_energy_db() -> None:
    """Copy energy.db to energy_v2.db (overwrites), without the *_metadata tables."""
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
    """Replace pv_gen columns of all PV players, scaled to the original total_energy."""
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
        if not pool:
            raise RuntimeError("runtime_pool is empty")

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

            factor = 0.8 # 1 # total_orig / total_new
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


def main():
    print("Ingesting load profiles")
    copy_energy_db()
    replace_base_load()
    replace_pv_gen()



if __name__ == "__main__":
    main()