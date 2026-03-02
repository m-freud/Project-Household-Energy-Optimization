import sqlite3
from src.config import Config

def create_sqlite_connection():
    return sqlite3.connect(Config.SQLITE_PATH)


def get_sqlite_cursor():
    return create_sqlite_connection().cursor()
