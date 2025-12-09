import sqlite3
import pandas as pd
import numpy as np
import os
import datetime as dt
from openpyxl import load_workbook
from openpyxl.utils import range_boundaries
from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write.point import Point
from influxdb_client.client.write_api import WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

from load.table_instructions import table_instructions


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


def period_to_epoch(period):
    seconds = period * 15 * 60
    base = dt.datetime(2000,1,1,0,0)
    ts = base + dt.timedelta(seconds=seconds)
    return ts


def load_to_influx(df, table_name):
    df["period"] = df["period"].apply(period_to_epoch)
    df = df.melt(
        id_vars=["period"],
        var_name="player_id",
        value_name="value").sort_values(by=["player_id", "period"]).reset_index(drop=True)

    write_api.write(
        bucket=bucket,
        record=df,
        data_frame_measurement_name=table_name,
        data_frame_field_name="value",
        data_frame_tag_columns=["player_id"],
        data_frame_timestamp_column="period")


def load_to_sqlite(df, table_name, config):
    process = config.get("process")
    if process:
        df = process(df)

    # create table schema first
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(config["schema"])

    # load data
    df.to_sql(table_name, conn, if_exists='append', index=False)


def load_table_to_db(wb, table_name, config):
    '''
    Loads a table from the Excel workbook into SQLite or InfluxDB based on the provided configuration.
    '''
    df = extract_df_from_xlsx(
        wb,
        config["sheet_name"],
        config["rectangle"],
        config["df_column_names"],
        config["transpose"]
    )

    if config.get("time_series"):
        load_to_influx(df, table_name)
    else:
        load_to_sqlite(df, table_name, config)


def load_all_tables(wb, table_instructions):
    for table_name, config in table_instructions.items():
        print(f"Loading table: {table_name}...")
        load_table_to_db(wb, table_name, config)
        print("done!")


# make these importable in other files:
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
excel_file_path = os.path.join(root_dir, "data", "A.xlsx")

# SQLite connection
db_path = os.path.join(root_dir, "db", "energy.db")

conn = sqlite3.connect(db_path)

# Influx connection
token = "97BMXHrBUuU--I2Wkm1KMqrePBEd-MI9fbyK9Ur8tkwoaeezJW6-x8rlXVjNB96HSZmqPaT89vnlU0GSroQ-fA=="
url = "http://localhost:8086"
org = "org"
bucket = "energy"

client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)
buckets_api = client.buckets_api()

if not buckets_api.find_bucket_by_name("energy"):
    buckets_api.create_bucket(bucket_name="energy", org=org)

# load workbook
wb = load_workbook(excel_file_path)

if __name__ == "__main__":
    wb = load_workbook(excel_file_path)

    for table_name, config in table_instructions.items():
        print(f"Loading table: {table_name}...")
        load_table_to_db(wb, table_name, config)

    print("done!")

    conn.close()