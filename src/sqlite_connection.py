import sqlite3


# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))
from src.config import Config

def create_sqlite_connection():
    return sqlite3.connect(Config.SQLITE_PATH)


def get_sqlite_cursor():
    return create_sqlite_connection().cursor()


sqlite_conn = create_sqlite_connection()
sqlite_cursor = get_sqlite_cursor()


def fetch_timeseries(sqlite_cursor, player_id, measurement):
    data = sqlite_cursor.execute(
        f'''
        SELECT period, "{player_id}" as value
        FROM {measurement}
        ORDER BY period
        ''',
    ).fetchall()

    return data


def fetch_multiple_timeseries(sqlite_cursor, player_id, measurements):
    timeseries_data = {}
    for measurement in measurements:
        data = fetch_timeseries(sqlite_cursor, player_id, measurement)
        timeseries_data[measurement] = data

    return timeseries_data


if __name__ == "__main__":
    # Example usage
    player_id = 12
    measurement = 'base_load'
    data = fetch_timeseries(sqlite_cursor, player_id, measurement)
    print(data)