import sqlite3
import pandas as pd

conn = sqlite3.connect("sqlite/staging/VISTA_ev1_status.db")
df = pd.read_sql_query("SELECT * FROM ev1_distances", conn).set_index("distance_type")

# Vergleicht die beiden Zeilen spaltenweise
unequal_persids = df.columns[df.loc["to_work_distance"] != df.loc["back_home_distance"]].tolist()

print(f"Anzahl ungleicher IDs: {len(unequal_persids)}")
print(unequal_persids)