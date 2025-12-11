# Table instructions for loading data from Excel sheets into sqlite and influxdb.
# TODO: get units right (kW, kWh, ?)
# TODO: some columns are broken. see ev1_data -> initial battery

TIME_SERIES_DEFAULT = {
    "rectangle": "A2:IQ97",
    "transpose": False,
    "df_column_names": ["period"] + [i for i in range(1, 251)],
    "time_series": True
}


table_instructions = {
    "player_pv_ess": { # TODO sort all entries like this
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
    "ev1": {
        "sheet_name": "EVs",
        "rectangle": "B6:IQ25",
        "df_column_names": [
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
            "price_at_public_charge_station_eur"],
        "schema": """
        CREATE TABLE IF NOT EXISTS ev1 (
            player_id INTEGER PRIMARY KEY,
            model_id INTEGER,
            capacity REAL,
            charge REAL,
            discharge REAL,
            efficiency REAL,
            initial_battery_level REAL,
            final_battery_level REAL,
            consumption_per_km REAL,
            departure_period INTEGER,
            arrival_period INTEGER,
            distance_km REAL,
            consumption REAL,
            morning_trip_duration_periods INTEGER,
            afternoon_trip_duration_periods INTEGER,
            trip_duration_periods INTEGER,
            charger_type INTEGER,
            public_charger INTEGER,
            power REAL,
            price_at_public_charge_station_eur REAL
        )""",
        "transpose": True,
        "process": None
    },
    "ev2": {
        "sheet_name": "EVs",
        "rectangle": "B27:IQ46",
        "df_column_names": [
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
            "price_at_public_charge_station_eur"],
        "schema": """
            CREATE TABLE IF NOT EXISTS ev2_data (
                player_id INTEGER PRIMARY KEY,
                model_id INTEGER,
                capacity REAL,
                charge REAL,
                discharge REAL,
                efficiency REAL,
                initial_battery_level REAL,
                final_battery_level REAL,
                consumption_per_km REAL,
                departure_period INTEGER,
                arrival_period INTEGER,
                distance_km REAL,
                consumption REAL,
                morning_trip_duration_periods INTEGER,
                afternoon_trip_duration_periods INTEGER,
                trip_duration_periods INTEGER,
                charger_type INTEGER,
                public_charger INTEGER,
                power REAL,
                price_at_public_charge_station_eur REAL
            )""",
        "transpose": True,
        "process": None
    },
    "load": {
        "sheet_name": "Load",
        **TIME_SERIES_DEFAULT
    },
    "pv": {
        "sheet_name": "PV",
        **TIME_SERIES_DEFAULT
    },
    "bess": {
        "sheet_name": "BESS",
        "rectangle": "B1:IQ8",
        "df_column_names": ["player_id", "model_id", "capacity", "charge", "discharge", "efficiency", "initial_soc", "final_soc"],
        "schema": """
        CREATE TABLE IF NOT EXISTS bess (
            player_id INTEGER PRIMARY KEY,
            model_id INTEGER,
            capacity REAL,
            charge REAL,
            discharge REAL,
            efficiency REAL,
            initial_soc REAL,
            final_soc REAL
        )""",
        "transpose": True,
        "process": None
    },
    "buy_price": {
        "sheet_name": "Buy Price",
        **TIME_SERIES_DEFAULT
    },
    "sell_price": {
        "sheet_name": "Sell Price",
        **TIME_SERIES_DEFAULT
    },
    "limits": {
        "sheet_name": "Limits",
        "rectangle": "B1:IQ10",
        "df_column_names": ["player_id", "power_buy", "power_sell", "fixed_costs_eur","_","initial_cp","premium_charger_edp_capacity","sum","new_cp_level","fixed_costs_2_eur"],
        "schema": """
        CREATE TABLE IF NOT EXISTS limits (
            player_id INTEGER PRIMARY KEY,
            power_buy REAL,
            power_sell REAL,
            fixed_costs_eur REAL,
            initial_cp REAL,
            premium_charger_edp_capacity REAL,
            sum REAL,
            new_cp_level REAL,
            fixed_costs_2_eur REAL
        )""",
        "transpose": True,
        "process": lambda df: df.drop(columns=["_"])
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