import sqlite3
import pandas as pd
import os
import openpyxl
from openpyxl import load_workbook
from openpyxl.utils import range_boundaries


def get_subtable_as_df(sheet_name, rectangle, column_names, flip_upright: bool = False): # TODO add flip option
    # get subtable of excel sheet as dataframe.
    data_path = os.path.join(os.getcwd(), "data", "A.xlsx")
    wb = load_workbook(data_path, data_only=True)[sheet_name]
    min_col, min_row, max_col, max_row = range_boundaries(rectangle)

    data = wb.iter_rows(min_row=min_row, max_row=max_row, min_col=min_col, max_col=max_col, values_only=True)
    data = list(data)

    df = pd.DataFrame(data)

    if flip_upright:
        df = df.T

    df.columns = column_names
    return df


subtables = {}

# "General Information" sheet

subtables["players"] = get_subtable_as_df(
    sheet_name="General Information",
    rectangle="A5:C254",
    column_names=["player_id", "pv", "ess"]
)

subtables["general_counts"] = get_subtable_as_df(
    sheet_name="General Information",
    rectangle="E4:F6",
    column_names=["unit", "count"]
)

subtables["contractual_power_terms"] = get_subtable_as_df(
    sheet_name="General Information",
    rectangle="E14:G24",
    column_names=["cp_kva", "price_per_day", "count"]
)

subtables["time_of_use_tariff_3"] = get_subtable_as_df(
    sheet_name="General Information",
    rectangle="E29:G31",
    column_names=["cp_kva", "leq_20_7", "gt_20_7"]
)

subtables["time_of_use_tariff_4"] = get_subtable_as_df(
    sheet_name="General Information",
    rectangle="E35:H37",
    column_names=["super_off_peak", "off_peak", "mid_peak", "on_peak"]
)

subtables["energy_storage_systems"] = get_subtable_as_df(
    sheet_name="General Information",
    rectangle="K5:Q12",
    column_names=["type", "capacity_kW", "charge_kW", "Discharge_kW", "efficiency_percent", "model", "units"]
)

subtables["premium_charger_edp"] = get_subtable_as_df(
    sheet_name="General Information",
    rectangle="K17:M18",
    column_names=["type", "capacity_kW", "units"]
)

subtables["regular_ev_models"] = get_subtable_as_df(
    sheet_name="General Information",
    rectangle="T5:AB19",
    column_names = ["model_id", "brand", "model_name", "capacity_kW", "charge_kW", "discharge_kW", "efficiency_percent", "consumption_Wh_km", "units"]
)

subtables["premium_ev_models"] = get_subtable_as_df(
    sheet_name="General Information",
    rectangle="T24:AB33",
    column_names = ["model_id", "brand", "model_name", "capacity_kW", "charge_kW", "discharge_kW", "efficiency_percent", "consumption_Wh_km", "units"]
)

subtables["ev_departures_arrivals"] = get_subtable_as_df(
    sheet_name="General Information",
    rectangle="AD6:AK33",
    column_names= ["hours",	"period", "departures_ev1", "arrivals_ev1", "departures_ev1", "arrivals_ev2", "departures_total", "arrivals_total"]
)

# "MOBIe EV chargers" sheet

subtables["mobie_ev_chargers"] = get_subtable_as_df(
    sheet_name="MOBIe EV chargers",
    rectangle="B3:O86",
    column_names=["id",	"uid_of_charger", "type_of_charging",	"state_of_charger",	"city",	"address", "operator", "capacity_kW", "voltage_level", "price_eur_per_charge",	"price_eur_per_minute",	"price_eur_per_kWh","price_eur_per_h",	"price_eur_per_kWh_again"]
) # TODO: double column € per kwh ??

# EVs sheet

# tables have to be flipped upright here
subtables["player_evs"] = get_subtable_as_df(
    sheet_name="EVs",
    rectangle="B1:IQ3",
    column_names=["id", "ev1_model", "ev2_model"],
    flip_upright=True
)

subtables["ev1_models"]


for table in subtables:
    print(f"{table}:\n{subtables[table]}\n")


# conn = sqlite3.connect('energy_test.db')
# cur = conn.cursor()

# cur.execute("""
# CREATE TABLE IF NOT EXISTS Players (
#     Player_ID INTEGER PRIMARY KEY,
#     has_PV INTEGER,
#     has_ESS INTEGER
# )
# """)


# players_df.to_sql('Players', conn, if_exists='replace', index=False)

# conn.commit()

# conn.close()