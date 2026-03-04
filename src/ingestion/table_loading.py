import pandas as pd
import numpy as np
import datetime as dt
from openpyxl import load_workbook
from openpyxl.utils import range_boundaries

# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))


from src.ingestion.table_config import table_config
from src.sqlite_connection import sqlite_conn
from src.config import Config



def extract_df_from_xlsx(wb, sheet_name, rectangle, column_names, transpose=False):
    '''
    Extracts data from an Excel worksheet and returns it as a pandas DataFrame.
    Parameters:
        wb: openpyxl Workbook object
        sheet_name: Name of the worksheet to extract data from
        rectangle: Cell range in A1 notation (e.g., "A1:C10")
        column_names: List of chosen column names for the DataFrame -> same as in sql table  (mostly the same but slugged, few exceptions)
        transpose: Boolean indicating whether to transpose the table
    '''
    ws = wb[sheet_name]
    min_col, min_row, max_col, max_row = range_boundaries(rectangle)

    rows = ws.iter_rows(
        min_row=min_row, max_row=max_row,
        min_col=min_col, max_col=max_col,
        values_only=True
    )

    data = list(rows)

    if transpose:
        data = np.array(data).T.tolist()

    return pd.DataFrame(data, columns=column_names)


def load_to_sqlite(df, table_name, config):
    # apply optional processing function
    process = config.get("process")
    if process:
        df = process(df)

    # drop table if exists
    sqlite_conn.execute(f"DROP TABLE IF EXISTS {table_name}")

    # load data
    df.to_sql(table_name, sqlite_conn, if_exists='append', index=False)


def load_table_to_db(wb, table_name, config):
    '''
    Loads a table from the Excel workbook into SQLite or InfluxDB based on the provided configuration.
    '''
    df = extract_df_from_xlsx(
        wb,
        config["sheet_name"],
        config["rectangle"],
        config["df_column_names"],
        config.get("transpose", False)
    )

    load_to_sqlite(df, table_name, config)


def load_all_tables(wb, table_instructions):
    for table_name, config in table_instructions.items():
        print(f"Loading table: {table_name} to sqlite...")
        load_table_to_db(wb, table_name, config)
    
    print("tables loaded successfully!")


if __name__ == "__main__":
    wb = load_workbook(Config.EXCEL_FILE_PATH, data_only=True)

    load_all_tables(wb, table_config)

    sqlite_conn.close()
