import sqlite3
import unittest

import pandas as pd

import src.ingestion.table_loading as table_loading


class AddCombinedEvStatesTest(unittest.TestCase):
    def setUp(self):
        self.original_sqlite_conn = table_loading.sqlite_conn
        self.conn = sqlite3.connect(":memory:")
        table_loading.sqlite_conn = self.conn

        for ev_num in ("1", "2"):
            self.conn.execute(f'CREATE TABLE ev{ev_num}_at_home (period INTEGER, "1" REAL, "2" REAL)')
            self.conn.execute(
                f'CREATE TABLE ev{ev_num}_at_charging_station (period INTEGER, "1" REAL, "2" REAL)'
            )

        self.conn.executemany(
            'INSERT INTO ev1_at_home VALUES (?, ?, ?)',
            [(1, 1.0, 0.0), (2, 0.0, 1.0)],
        )
        self.conn.executemany(
            'INSERT INTO ev1_at_charging_station VALUES (?, ?, ?)',
            [(1, 0.0, 1.0), (2, 1.0, 0.0)],
        )
        self.conn.executemany(
            'INSERT INTO ev2_at_home VALUES (?, ?, ?)',
            [(1, 0.0, 1.0), (2, 1.0, 0.0)],
        )
        self.conn.executemany(
            'INSERT INTO ev2_at_charging_station VALUES (?, ?, ?)',
            [(1, 1.0, 0.0), (2, 0.0, 1.0)],
        )
        self.conn.commit()

    def tearDown(self):
        table_loading.sqlite_conn = self.original_sqlite_conn
        self.conn.close()

    def test_add_combined_ev_states_builds_wide_status_tables(self):
        table_loading.add_combined_ev_states()

        ev1_status = pd.read_sql_query('SELECT * FROM ev1_status ORDER BY period', self.conn)
        ev2_status = pd.read_sql_query('SELECT * FROM ev2_status ORDER BY period', self.conn)

        self.assertEqual(ev1_status.columns.tolist(), ["period", "1", "2"])
        self.assertEqual(ev2_status.columns.tolist(), ["period", "1", "2"])
        self.assertEqual(ev1_status[["1", "2"]].values.tolist(), [[0.0, 2.0], [2.0, 0.0]])
        self.assertEqual(ev2_status[["1", "2"]].values.tolist(), [[2.0, 0.0], [0.0, 2.0]])


if __name__ == "__main__":
    unittest.main()