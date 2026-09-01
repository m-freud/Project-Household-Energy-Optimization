# load parqueet file and print shape

import pandas as pd
from pathlib import Path

p_path = Path(__file__).parent

df = pd.read_parquet(p_path / "load_features_01-2023.parquet")
print(df.shape)