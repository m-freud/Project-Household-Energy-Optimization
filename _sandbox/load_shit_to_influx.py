from influxdb_client.client.influxdb_client import InfluxDBClient
from influxdb_client.client.write.point import Point
from influxdb_client.client.write_api import WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS
import datetime as dt
import pandas as pd


token = "97BMXHrBUuU--I2Wkm1KMqrePBEd-MI9fbyK9Ur8tkwoaeezJW6-x8rlXVjNB96HSZmqPaT89vnlU0GSroQ-fA=="
url = "http://localhost:8086"
org = "org"
bucket = "energy"


client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)

# --- FAKE DF (your head) ---
df = pd.DataFrame({
    "player_id": [1, 1, 2],
    "period": [0, 1, 2],              # periods
    "value": [10.5, 11, 12]       # fields
})

# --- PERIOD → TIMESTAMP ---
def period_to_ts(p):
    base = dt.datetime(2025, 1, 1)
    return base + dt.timedelta(minutes=p * 15)

df["period"] = df["period"].apply(period_to_ts)

print(df)

write_api.write(
    bucket=bucket,
    record=df,
    data_frame_measurement_name="load",
    data_frame_field_name="value",
    data_frame_tag_columns=["player_id"],
    data_frame_timestamp_column="period"
    )

print("🔥 Loaded 3 points into Influx!")