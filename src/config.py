from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

class Config:
    ROOT_DIR = Path(__file__).parent.parent

    # data source
    EXCEL_FILE_PATH = ROOT_DIR / "data" / "A.xlsx"

    # sqlite
    SQLITE_PATH = ROOT_DIR / "sqlite" / "energy.db"
