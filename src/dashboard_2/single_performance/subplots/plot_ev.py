import matplotlib.pyplot as plt
from src.simulation.scenarios.scenario import get_scenario_value
from src.sqlite_connection import load_series

from src.dashboard_2.single_performance.subplots.helpers import shade_ev_location_background



def plot_ev(ax: plt.Axes,
            ev_number: str,
            scenario_name: str,
            player_id: int,
            policy_colors: dict[str, str]
            ) -> None:
    ax.set_title(f"EV{ev_number}")
    ax.set_ylabel("SOC (kWh)")
    ev_at_home_df = load_series(f"ev{ev_number}_at_home", player_id)
    ev_at_station_df = load_series(f"ev{ev_number}_at_charging_station", player_id)
    shade_ev_location_background(ax, ev_at_home_df, ev_at_station_df)

    target_soc_ev = get_scenario_value(scenario_name, f"ev{ev_number}", player_id, "target_soc")
    ev_deadline = get_scenario_value(scenario_name, f"ev{ev_number}", player_id, "deadline")

    if target_soc_ev is not None:
        ax.axhline(
        y=target_soc_ev,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label="Target SOC",
    )
        
    for policy_name in policy_colors.keys():
        color = policy_colors[policy_name]
        ev_soc_df = load_series(f"ev{ev_number}_soc", player_id, scenario_name, policy_name)

        if ev_soc_df.empty:
            continue

        ax.plot(ev_soc_df["hour"], ev_soc_df["value"], color=color, linewidth=2)

    if ev_deadline is not None:
        ev_deadline_hour = float(ev_deadline) / 4.0
        ax.axvline(x=ev_deadline_hour, color="darkblue", linewidth=1.8)
