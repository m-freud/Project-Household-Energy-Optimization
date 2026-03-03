"""
This file contains specialized functions to plot household data.
Different functions focus on different aspects of household day history/performance

"""

# paste this to enable src. imports

from pathlib import Path
import sys
import matplotlib.pyplot as plt
from datetime import datetime

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.scenarios.scenario import default_scenario
from src.simulation.policies.naive_linear_satisfaction import naive_linear_satisfaction, last_minute_satisfaction


from src.sql_connection import get_sqlite_cursor
sqlite_cursor = get_sqlite_cursor()


from src.config import Config


default_colors = {
    "base_load": "blue",
    "pv_gen": "orange",
    "net_load": "red",
    "bess_power": "purple",
    "bess_soc": "green",
    "ev1_power": "brown",
    "ev1_soc": "pink",
    "ev2_power": "gray",
    "ev2_soc": "olive",
}


def get_household_info(household_id, table, field):
    '''
    get specific field from sqlite
    '''
    sqlite_cursor.execute(
        f"SELECT {field} FROM {table} WHERE player_id = ?", (household_id,)
    )

    result = sqlite_cursor.fetchone()
    return result[0] if result else None


def datetime_parser(dt_str):
    # convert from datetime object to hh:mm format
    if isinstance(dt_str, datetime):
        return dt_str.strftime("%H:%M")


def plot_household_fields(
        household_id,
        policy= naive_linear_satisfaction,
        scenario= default_scenario,
        fields={"base_load", "pv_gen", "net_load"},
        colors= default_colors,
        title="",
        figsize=(12, 6)
        ):
    '''
    plot specified fields for a household given a policy and scenario
    '''
    
    # Build Flux query
    flux_set = "[" + ",".join(f'"{f}"' for f in fields) + "]"
    query = f'''
    from(bucket: "{Config.INFLUX_BUCKET}")
        |> range(start: 0)
        |> filter(fn: (r) => r["player_id"] == "{household_id}")
        |> filter(fn: (r) => r["policy"] == "{policy.__name__}" or not exists r["policy"]) // base inputs dont have a policy tag
        |> filter(fn: (r) => r["scenario"] == "{scenario.name}" or not exists r["scenario"]) // base inputs dont have a scenario tag
        |> filter(fn: (r) => contains(value: r["_measurement"], set: {flux_set}))
        |> sort(columns: ["_time"])
    '''

    result = influx_query_api.query(org=Config.INFLUX_ORG, query=query)

    if not result:
        raise ValueError("No data returned from InfluxDB for the given parameters.")
    
    # Extract data
    data = {}
    for table in result:
        if not table.records:
            continue
        measurement = table.records[0].values['_measurement']
        times = [datetime_parser(record.get_time()) for record in table.records]
        values = [record.get_value() for record in table.records]
        data[measurement] = {'times': times, 'values': values}
    
    if not data:
        raise ValueError(f"No data found for fields {fields} in household {household_id} with policy {policy.__name__} and scenario {scenario.name}")
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each requested field
    for field in fields:
        if field not in data:
            print(f"Warning: No data found for field '{field}'")
            continue

        times = data[field]['times']
        values = data[field]['values']

        # If times are strings (HH:MM), plot against integer indices and set labels later
        if times and isinstance(times[0], str):
            x = list(range(len(times)))
            ax.plot(x, values, label=field, color=default_colors.get(field, None), linewidth=1.5)
        else:
            # assume datetime-like or numeric x-values which matplotlib can handle
            ax.plot(times, values, label=field, color=default_colors.get(field, None), linewidth=1.5)

    # Add horizontal target_soc lines for each device
    bess_capacity = get_household_info(household_id, table="bess", field="capacity")
    if bess_capacity is not None and "bess_soc" in fields:
        target_soc = scenario.bess.target_soc * bess_capacity
        ax.axhline(y=target_soc, color='green', linestyle='--', label='BESS Target SOC')
    
    ev1_capacity = get_household_info(household_id, table="ev1", field="capacity")
    if ev1_capacity is not None and "ev1_soc" in fields:
        target_soc_ev1 = scenario.ev1.target_soc * ev1_capacity
        ax.axhline(y=target_soc_ev1, color='purple', linestyle='--', label='EV1 Target SOC')

    ev2_capacity = get_household_info(household_id, table="ev2", field="capacity")
    if ev2_capacity is not None and "ev2_soc" in fields:
        target_soc_ev2 = scenario.ev2.target_soc * ev2_capacity
        ax.axhline(y=target_soc_ev2, color='brown', linestyle='--', label='EV2 Target SOC')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(title or f'Household {household_id} with policy {policy.__name__} and scenario {scenario.name}')
    # Configure x-axis ticks: prefer fixed hourly ticks at 0,4,8,12,16,20,24 (hours)
    # For 15-min steps each hour = 4 steps -> positions = hour * 4
    sample_times = None
    for v in data.values():
        if v.get('times'):
            sample_times = v['times']
            break

    if sample_times:
        first = sample_times[0]
        if isinstance(first, str):
            N = len(sample_times)
            hours = [0, 4, 8, 12, 16, 20, 24]
            xticks = [h * 4 for h in hours if h * 4 <= (N - 1)]
            if xticks:
                xtick_labels = [f"{h:02d}:00" for h in hours if h * 4 <= (N - 1)]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
        elif isinstance(first, datetime):
            import matplotlib.dates as mdates
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            fig.autofmt_xdate()
        else:
            try:
                max_step = max(int(float(t)) for t in sample_times)
                hours = [0, 4, 8, 12, 16, 20, 24]
                xticks = [h * 4 for h in hours if h * 4 <= max_step]
                if xticks:
                    xtick_labels = [f"{h:02d}:00" for h in hours if h * 4 <= max_step]
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
            except Exception:
                pass

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_household_devices(
    household_id: int,
    policy= last_minute_satisfaction,
    scenario= default_scenario,
    colors = default_colors,
    figsize = (12, 6)
    ):
    
    fields = [
        "bess_soc",
        "ev1_soc",
        "ev2_soc",
    ]
    
    return plot_household_fields(
        household_id=household_id,
        policy=policy,
        scenario=scenario,
        colors=colors,
        fields=fields,
        figsize=figsize
    )


def plot_household_loads(
    household_id: int,
    policy= naive_linear_satisfaction,
    scenario= default_scenario,
    colors= default_colors,
    figsize = (12, 6)
):
    fields = [
        "base_load",
        "pv_gen",
        "net_load",
    ]
    
    return plot_household_fields(
        household_id=household_id,
        policy=policy,
        scenario=scenario,
        fields=fields,
        colors=colors,
        figsize=figsize
    )


def plot_household_costs(
    household_id: int,
    policy= naive_linear_satisfaction,
    scenario= default_scenario,
    colors= default_colors,
    figsize = (12, 6)
):
    fields = [
        "total_cost",
        "net_load",
        "buy_price",
    ]
    
    return plot_household_fields(
        household_id=household_id,
        policy=policy,
        scenario=scenario,
        fields=fields,
        colors=colors,
        figsize=figsize
    )


if __name__ == "__main__":
    # Example usage
    fields = ["base_load", "pv_gen", "net_load", "bess_soc", "ev1_soc", "ev2_soc"]
    fig = plot_household_devices(
        household_id=1,
        policy=last_minute_satisfaction,
        scenario=default_scenario,
    )