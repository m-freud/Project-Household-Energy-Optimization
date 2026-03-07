from src.sqlite_connection import load_series
import matplotlib.pyplot as plt

def plot_net_load(
        ax: plt.Axes,
        scenario_name: str,
        player_id: int,
        policy_colors: dict
        ) -> None:
    ax.set_title("Net Load")
    ax.set_ylabel("Power (kW)")
    ax.set_xlabel("Hour")
    
    for policy_name in policy_colors.keys():
        color = policy_colors[policy_name]
        net_load_df = load_series("net_load", player_id, scenario_name, policy_name)

        if net_load_df.empty:
            continue

        ax.plot(net_load_df["hour"], net_load_df["value"], color=color, linewidth=2)
        ax.fill_between(net_load_df["hour"], net_load_df["value"], color=color, alpha=0.2)
    
    ax.axhline(y=0.0, color="black", linewidth=1, alpha=0.5)
    ax.legend(loc="upper left")
