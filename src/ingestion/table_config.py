# Table instructions for loading data from Excel sheets into sqlite and influxdb.

EV_COLUMNS = [
    "player_id",
    "model_id",
    "capacity",
    "charge",
    "discharge",
    "efficiency",
    "initial_battery_level",
    "final_battery_level",
    "consumption_per_km",
    "departure_period",
    "arrival_period",
    "distance_km",
    "consumption",
    "morning_trip_duration_periods",
    "afternoon_trip_duration_periods",
    "trip_duration_periods",
    "charger_type",
    "public_charger",
    "power",
    "price_at_public_charge_station_eur"
]


TIME_SERIES_DEFAULT = {
    "rectangle": "A2:IQ97",
    "transpose": False,
    "df_column_names": ["period"] + [i for i in range(1, 251)],
    "time_series": True
}


table_instructions = {
    "player_pv_ess": {
        "sheet_name": "General Information",
        "rectangle": "A5:C254",
        "transpose": False,
        "df_column_names": ["player_id", "has_pv", "has_ess"],
        "process": None,
        "time_series": False,
        "schema": """
        CREATE TABLE IF NOT EXISTS player_pv_ess (
            player_id INTEGER PRIMARY KEY,
            has_pv Boolean,
            has_ess Boolean
        )""",
    },
    "bess": {
        "sheet_name": "BESS",
        "rectangle": "B1:IQ8",
        "df_column_names": ["player_id", "model_id", "capacity", "charge", "discharge", "efficiency", "initial_soc", "final_soc"],
        "schema": "",
        "transpose": True,
        "process": lambda df: df.assign(model_id=lambda x: x.model_id.astype(int)) # ensure model_id is int for aesthetic reasons
    },
    "ev1": {
        "sheet_name": "EVs",
        "rectangle": "B6:IQ25",
        "df_column_names": EV_COLUMNS,
        "schema": "",
        "transpose": True,
        "process": None
    },
    "ev2": {
        "sheet_name": "EVs",
        "rectangle": "B27:IQ46",
        "df_column_names": EV_COLUMNS,
        "schema": "",
        "transpose": True,
        "process": None
    },
    "fixed_costs": {
        "sheet_name": "Limits",
        "rectangle": "B1:IQ10",
        "df_column_names": ["player_id", "power_buy", "power_sell", "fixed_costs", "_", "initial_cp", "premium_charger_edp_capacity","sum","new_cp_level","fixed_costs_2_eur"],
        "schema": "",
        "transpose": True,
        "process": lambda df: df[["player_id", "fixed_costs"]] # so far we only need fixed costs, if anything. reexpand if you want to change limits
    },
    "load": {
        "sheet_name": "Load",
        **TIME_SERIES_DEFAULT
    },
    "pv": {
        "sheet_name": "PV",
        **TIME_SERIES_DEFAULT
    },
    "buy_price": {
        "sheet_name": "Buy Price",
        **TIME_SERIES_DEFAULT
    },
    "sell_price": {
        "sheet_name": "Sell Price",
        **TIME_SERIES_DEFAULT
    },
    "ev1_buy_price": {
        "sheet_name": "EV1 Buy Price",
        **TIME_SERIES_DEFAULT
    },
    "ev2_buy_price": {
        "sheet_name": "EV2 Buy Price",
        **TIME_SERIES_DEFAULT
    },
    "ev1_load": {
        "sheet_name": "EV1 Load",
        **TIME_SERIES_DEFAULT
    },
    "ev2_load": {
        "sheet_name": "EV2 Load",
        **TIME_SERIES_DEFAULT
    },
    "ev1_max_discharge": {
        "sheet_name": "Max EV1 Dis-Charge",
        **TIME_SERIES_DEFAULT
    },
    "ev2_max_discharge": {
        "sheet_name": "Max EV2 Dis-Charge",
        **TIME_SERIES_DEFAULT
    },
    "ev1_at_home": {
        "sheet_name": "EV1 at Home (x)",
        **TIME_SERIES_DEFAULT
    },
    "ev2_at_home": {
        "sheet_name": "EV2 at Home (x)",
        **TIME_SERIES_DEFAULT
    },
    "ev1_at_charging_station": {
        "sheet_name": "EV1 at Charging Station (x)",
        **TIME_SERIES_DEFAULT
    },
    "ev2_at_charging_station": {
        "sheet_name": "EV2 at Charging Station (x)",
        **TIME_SERIES_DEFAULT
    }
}