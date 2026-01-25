"""
Household Plotter - Query and visualize household energy data from InfluxDB.

Decoupled from simulation - queries data directly from InfluxDB.
"""

import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from src.connections import get_influx_query_api
from src.config import Config


def plot_household(
    household_id: int,
    policy: str,
    fields: List[str],
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
        |> filter(fn: (r) => r["policy"] == "{policy}" or not exists r["policy"]) // base inputs dont have a policy tag
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
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(title or f'Household {household_id} with policy {policy}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig
