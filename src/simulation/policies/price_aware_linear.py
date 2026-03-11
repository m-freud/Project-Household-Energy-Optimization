import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))
from src.config import Config
