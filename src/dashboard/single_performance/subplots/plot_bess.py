from src.simulation.scenarios.scenario import get_scenario_value
from src.sqlite_connection import load_series
import matplotlib.pyplot as plt



def plot_bess(
        ax: plt.Axes,
        scenario_name: str,
        player_id: int,
    policy_colors: dict[str, str],
    missed_deadline: bool = False,
    ) -> None:
    ax.set_title("BESS")
    ax.set_ylabel("SOC (kWh)")

    target_soc_bess = get_scenario_value(scenario_name, "bess", player_id, "target_soc")
    bess_deadline = get_scenario_value(scenario_name, "bess", player_id, "deadline")

    if target_soc_bess is not None:
        ax.axhline(
        y=target_soc_bess,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label="Target SOC",
    )

    has_data = False
        
    for policy_name in policy_colors.keys():
        color = policy_colors[policy_name]
        bess_soc_df = load_series("bess_soc", player_id, scenario_name, policy_name)

        if bess_soc_df.empty:
            continue

        has_data = True

        ax.plot(bess_soc_df["hour"], bess_soc_df["value"], color=color, linewidth=2)

    if not has_data:
        ax.text(0.5, 0.5, "No BESS SOC data", transform=ax.transAxes, ha="center", va="center")

    if bess_deadline is not None:
        bess_deadline_hour = float(bess_deadline) / 4.0
        ax.axvline(x=bess_deadline_hour, color="darkblue", linewidth=1.8)
        if missed_deadline:
            ax.axvline(x=bess_deadline_hour + 0.03, color="red", linewidth=1.8)
