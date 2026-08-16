import matplotlib.pyplot as plt
from src.sqlite_connection import load_series

def plot_base_load(
        ax: plt.Axes,
        scenario_name: str,
        player_id: int,
        policy_colors: dict[str, str]
        ) -> None:
    ax.set_title("Base Load")
    ax.set_ylabel("Load")
    ax.set_xlabel("Hour")

    has_data = False

    for policy_name in policy_colors.keys():
        color = policy_colors[policy_name]
        base_load_df = load_series("base_load", player_id, scenario_name, policy_name)

        if base_load_df.empty:
            continue

        has_data = True

        ax.plot(base_load_df["hour"], base_load_df["value"], color=color, linewidth=2)
        ax.fill_between(base_load_df["hour"], base_load_df["value"], color=color, alpha=0.2)

    if not has_data:
        ax.text(0.5, 0.5, "No base load data", transform=ax.transAxes, ha="center", va="center")

    ax.axhline(y=0.0, color="black", linewidth=1, alpha=0.5)
