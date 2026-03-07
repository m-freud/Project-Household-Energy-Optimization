import matplotlib.pyplot as plt
from src.simulation.scenarios.scenario import get_scenario_value
from src.sqlite_connection import load_series

def plot_pv(
        ax: plt.Axes,
        player_id: int):
    ax.set_title("PV Generation")
    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Hour")
    
    pv_gen_df = load_series("pv_gen", player_id)

    if pv_gen_df.empty:
        ax.text(0.5, 0.5, "No PV data", transform=ax.transAxes, ha="center", va="center")
        return

    ax.plot(pv_gen_df["hour"], pv_gen_df["value"], color="tab:orange", linewidth=2)
    ax.fill_between(pv_gen_df["hour"], pv_gen_df["value"], color="tab:orange", alpha=0.2)
    ax.legend(loc="upper left")
