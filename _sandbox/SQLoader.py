import sqlite3
import pandas as pd
import os

data_path = os.path.join(os.getcwd(), "data", "A.xlsx")

xls = pd.ExcelFile(data_path)
print(xls.sheet_names)


