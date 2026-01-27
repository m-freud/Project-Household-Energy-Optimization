"""
Household Plotter - Query and visualize household energy data from InfluxDB.

Decoupled from simulation - queries data directly from InfluxDB.
"""

# paste this to enable src. imports

from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from src.simulation.policies.naive_linear_satisfaction import naive_linear_satisfaction
from src.connections import get_influx_query_api
from src.config import Config
from src.simulation.scenarios.scenario import default_scenario
from src.connections import get_sqlite_cursor


sqlite_cursor = get_sqlite_cursor()

def get_household_info(household_id, table, field):
    sqlite_cursor.execute(
        f"SELECT {field} FROM {table} WHERE player_id = ?", (household_id,)
    )

    result = sqlite_cursor.fetchone()
    return result[0] if result else None


def plot_household_fields(
    household_id: int,
    policy= naive_linear_satisfaction,
    scenario= default_scenario,
    fields: List[str]=["base_load", "pv_gen"],
    colors: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 6)
):
    """
    Query and plot specified fields for a single household from InfluxDB.
    
    Parameters:
    -----------
    household_id : int
        The household ID to query
    policy : str
        The policy name to query
    fields : List[str]
        List of measurement names to plot (e.g., ['net_load', 'pv_generation'])
    colors : Dict[str, str], optional
        Mapping of field names to colors (e.g., {'net_load': 'blue', 'pv_generation': 'orange'})
    title : str, optional
        Custom plot title
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    
    Raises:
    -------
    ValueError
        If no data found for the household or specified fields
    """
    if not fields:
        raise ValueError("Must specify at least one field to plot")
    
    colors = colors or {}
    query_api = get_influx_query_api()
    
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
    
    result = query_api.query(org=Config.INFLUX_ORG, query=query)
    
    if not result:
        raise ValueError(f"No data found for household {household_id} with policy {policy}")
    
    # Extract data
    data = {}
    for table in result:
        if not table.records:
            continue
        measurement = table.records[0].values['_measurement']
        times = [record.get_time() for record in table.records]
        values = [record.get_value() for record in table.records]
        data[measurement] = {'times': times, 'values': values}
    
    if not data:
        raise ValueError(f"No data found for fields {fields} in household {household_id} with policy {policy}")
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    for field in fields:
        if field not in data:
            print(f"Warning: No data found for field '{field}'")
            continue
        
        color = colors.get(field)
        ax.plot(data[field]['times'], data[field]['values'], 
                label=field, color=color, linewidth=1.5)
    
    # Add horizontal target_soc lines for each device
    bess_capacity = get_household_info(household_id, table="bess", field="capacity")
    if bess_capacity is not None:
        target_soc = scenario.bess.target_soc * bess_capacity
        ax.axhline(y=target_soc, color='green', linestyle='--', label='BESS Target SOC')
    
    ev1_capacity = get_household_info(household_id, table="ev1", field="capacity")
    if ev1_capacity is not None:
        target_soc_ev1 = scenario.ev1.target_soc * ev1_capacity
        ax.axhline(y=target_soc_ev1, color='purple', linestyle='--', label='EV1 Target SOC')

    ev2_capacity = get_household_info(household_id, table="ev2", field="capacity")
    if ev2_capacity is not None:
        target_soc_ev2 = scenario.ev2.target_soc * ev2_capacity
        ax.axhline(y=target_soc_ev2, color='brown', linestyle='--', label='EV2 Target SOC')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(title or f'Household {household_id} with policy {policy}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_household_full(
    household_id: int,
    policy= naive_linear_satisfaction,
    scenario= default_scenario,
    figsize: tuple = (12, 8)
):
    """
    Plot a comprehensive set of household fields.
    
    Parameters:
    -----------
    household_id : int
        The household ID to query
    policy : str
        The policy name to query
    scenario : Scenario
        The scenario configuration
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    fields = [
        "base_load",
        "pv_gen",
        "net_load",
        "bess_power",
        "bess_soc",
        "ev1_power",
        "ev1_soc",
        "ev2_power",
        "ev2_soc",
    ]
    
    return plot_household_fields(
        household_id=household_id,
        policy=policy,
        scenario=scenario,
        fields=fields,
        figsize=figsize
    )

if __name__ == "__main__":
    max_charge_bess_h1 = get_household_info(household_id=1, table="bess", field="capacity")
    print(f"Max Charge BESS for Household 1: {max_charge_bess_h1}")