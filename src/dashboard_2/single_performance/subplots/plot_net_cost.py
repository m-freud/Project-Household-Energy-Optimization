import matplotlib.pyplot as plt
from src.simulation.scenarios.scenario import get_scenario_value
from src.sqlite_connection import load_series

def plot_net_cost(
        ax: plt.Axes,
        scenario_name: str,
        player_id: int,
        policy_colors: dict[str, str]
        ) -> None:
    ax.set_title("Net Cost")
    ax.set_ylabel("Cost (€)")
    ax.set_xlabel("Hour")
    
    for policy_name in policy_colors.keys():
        color = policy_colors[policy_name]
        net_cost_df = load_series("net_cost", player_id, scenario_name, policy_name)

        if net_cost_df.empty:
            continue

        ax.plot(net_cost_df["hour"], net_cost_df["value"], color=color, linewidth=2)
        ax.fill_between(net_cost_df["hour"], net_cost_df["value"], color=color, alpha=0.2)
    
    ax.axhline(y=0.0, color="black", linewidth=1, alpha=0.5)
    ax.legend(loc="upper left")
