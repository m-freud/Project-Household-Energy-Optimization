import sqlite3
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
DE_PV_DB = ROOT_DIR / "sqlite" / "staging" / "DE_pv_gen.db"
SOURCE_PLAYER = "residential3"
SOURCE_YEAR = 2018


def create_runtime_pool() -> None:
    """runtime_pool: all pv_gen_pool days of SOURCE_PLAYER in SOURCE_YEAR, chronological."""
    conn = sqlite3.connect(DE_PV_DB)
    try:
        conn.execute("DROP TABLE IF EXISTS runtime_pool")
        conn.execute(
            "CREATE TABLE runtime_pool AS "
            "SELECT * FROM pv_gen_pool WHERE household_id = ? AND date LIKE ? ORDER BY date",
            (SOURCE_PLAYER, f"{SOURCE_YEAR}-%"),
        )
        conn.commit()
        print("runtime_pool rows:", conn.execute("SELECT COUNT(*) FROM runtime_pool").fetchone()[0])
    finally:
        conn.close()


def main():
    create_runtime_pool()


if __name__ == "__main__":
    main()