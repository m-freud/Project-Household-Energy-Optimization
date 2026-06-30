"""
Compare new MPC controller against run_id=5 baseline.
Runs households 1+2, scenarios default_scenario + low_start_wide,
then prints a diff table.
"""
import sys
import time
import sqlite3

from pathlib import Path
repo_root = next((p for p in Path(__file__).resolve().parents if (p / "src").exists()), Path(__file__).parent)
sys.path.insert(0, str(repo_root))

from src.sqlite_connection import sqlite_conn
from src.simulation.simulation import Simulation, RunContext, make_mpc_controller
from src.simulation.scenarios.scenario import scenarios as scenario_catalog

TARGET_HOUSEHOLDS = [1, 2]
TARGET_SCENARIOS  = ["default_scenario", "low_start_wide"]

# ── pull baseline ────────────────────────────────────────────────────────────
local_conn = sqlite3.connect(repo_root / "sqlite" / "energy.db")
cur = local_conn.cursor()
baseline = {}
rows = cur.execute("""
    SELECT player_id, scenario, total_cost, total_consumption, net_cost, net_load
    FROM results
    WHERE run_id = '5'
      AND player_id IN (1,2)
      AND scenario IN ('default_scenario','low_start_wide')
""").fetchall()
for hid, scen, tc, tcon, nc, nl in rows:
    baseline[(hid, scen)] = dict(total_cost=tc, total_consumption=tcon, net_cost=nc, net_load=nl)
local_conn.close()

print(f"\nBaseline rows found: {len(baseline)}")

# ── run new controller ───────────────────────────────────────────────────────
scenarios_by_name = {s.name: s for s in scenario_catalog}
sim = Simulation(sqlite_conn, ensure_schema=False)
new_results = {}

for scen_name in TARGET_SCENARIOS:
    scenario = scenarios_by_name[scen_name]
    for hid in TARGET_HOUSEHOLDS:
        print(f"\nRunning h={hid}  scenario={scen_name} ...", flush=True)
        t0 = time.perf_counter()
        rc = RunContext(
            controller_factory=make_mpc_controller("mpc_oracle", horizon=96),
            controller_name="mpc_oracle",
            scenario=scenario,
            start_time=1,
        )
        h = sim.create_household(hid, rc)
        c = sim.create_controller(h, rc)
        for t in range(1, 97):
            sim.step(h, c, scenario, time=t)
        elapsed = time.perf_counter() - t0
        new_results[(hid, scen_name)] = dict(
            total_cost=h.total_cost,
            total_consumption=h.total_consumption,
            net_cost=sum(h.history["net_cost"].values()) * 0.25,
            net_load=sum(h.history["net_load"].values()) * 0.25,
        )
        print(f"  done in {elapsed:.1f}s  total_cost={h.total_cost:.4f}  total_consumption={h.total_consumption:.4f}")

# ── comparison table ─────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print(f"{'HH':<4} {'Scenario':<22} {'Metric':<18} {'Baseline':>12} {'New':>12} {'Delta':>12} {'Delta%':>8}")
print("=" * 90)

for (hid, scen) in sorted(new_results.keys()):
    b = baseline.get((hid, scen))
    n = new_results[(hid, scen)]
    if b is None:
        print(f"{hid:<4} {scen:<22}  *** no baseline row ***")
        continue
    for metric in ["total_cost", "total_consumption", "net_cost", "net_load"]:
        bv, nv = b[metric], n[metric]
        delta = nv - bv
        pct   = (delta / bv * 100) if bv else float("nan")
        flag  = " !" if abs(pct) > 0.5 else ""
        print(f"{hid:<4} {scen:<22} {metric:<18} {bv:>12.4f} {nv:>12.4f} {delta:>+12.4f} {pct:>7.2f}%{flag}")
    print("-" * 90)

print("\nDone.")
