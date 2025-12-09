from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
    INFLUX_URL = os.getenv("INFLUX_URL")
    INFLUX_ORG = os.getenv("INFLUX_ORG")
    INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")