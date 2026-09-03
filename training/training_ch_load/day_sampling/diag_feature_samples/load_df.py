# load parqueet file and print shape

import pandas as pd
from pathlib import Path

p_path = Path(__file__).parent

df = pd.read_parquet(p_path / "diag_100707_features.parquet")
print(df.shape)
print(df)