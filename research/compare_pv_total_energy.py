import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

with sqlite3.connect("sqlite/energy.db") as c:
    orig = pd.read_sql_query("SELECT total_energy FROM pv_gen_metadata", c)["total_energy"]
with sqlite3.connect("sqlite/staging/DE_pv_gen.db") as c:
    new = pd.read_sql_query("SELECT total_energy FROM pv_gen_metadata", c)["total_energy"]

print(pd.DataFrame({"energy.db": orig.describe(), "DE_pv_gen.db": new.describe()}))
print("zeros:", (orig == 0).sum(), (new == 0).sum())

rng = (min(orig.min(), new.min()), max(orig.max(), new.max()))
plt.hist(orig, bins=40, range=rng, alpha=0.5, label=f"energy.db (n={len(orig)})", density=True)
plt.hist(new, bins=40, range=rng, alpha=0.5, label=f"DE_pv_gen.db (n={len(new)})", density=True)
plt.xlabel("total pv generation (total_energy)")
plt.ylabel("density")
plt.legend()
plt.tight_layout()
plt.savefig("plots/pv_total_energy_hist.png", dpi=150)
