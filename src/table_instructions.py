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
            has_pv INTEGER,
            has_ess INTEGER
        )""",
    },
    "total": {
        "sheet_name": "General Information",
        "rectangle": "E4:F6",
        "transpose": False,
        "df_column_names": ["unit", "count"],
        "process": None,
        "time_series": False,
        "schema": """
        CREATE TABLE IF NOT EXISTS total (
            unit TEXT PRIMARY KEY,
            count INTEGER
        )""",
    },
    "contractual_power_terms": {
        "sheet_name": "General Information",
        "rectangle": "E14:G24",
        "df_column_names": ["cp_kva", "price_per_day", "count"],
        "schema": """
        CREATE TABLE IF NOT EXISTS contractual_power_terms (
            cp_kva REAL PRIMARY KEY,
            price_per_day REAL,
            count INTEGER
        )""",
        "transpose": False,
        "process": None
    },
    "time_of_use_tariff_3": {
        "sheet_name": "General Information",
        "rectangle": "F28:G31",
        "df_column_names": ["cp_kva", "on_peak_eur", "mid_peak_eur", "off_peak_eur"],
        "schema": """
        CREATE TABLE IF NOT EXISTS time_of_use_tariff_3 (
            cp_kva TEXT PRIMARY KEY,
            on_peak_eur REAL,
            mid_peak_eur REAL,
            off_peak_eur REAL
        )""",
        "transpose": True,
        "process": lambda df: df.assign(
            cp_kva=df['cp_kva'].replace({
                "<=20,7": "leq_20_7",
                ">20,7": "gt_20_7"
            })
        )
    },
    # "time_of_use_tariff_4" : {    ????
    #     "sheet_name": "General Information",
    #     "rectangle": "E35:H37",
    #     "df_column_names": ["super_off_peak", "off_peak", "mid_peak", "on_peak"],
    #     "schema": ?
    #     "transpose": False,
    #     "process": """transform into table period vs peak_type. Where are the prices?"""
    # },
    "energy_storage_system_models": {
        "sheet_name": "General Information",
        "rectangle": "K5:Q12",
        "df_column_names": ["type", "capacity_kW", "charge_kW", "Discharge_kW", "efficiency_percent", "model", "units"],
        "schema": """
        CREATE TABLE IF NOT EXISTS energy_storage_system_models (
            type INTEGER PRIMARY KEY,
            capacity_kW REAL,
            charge_kW REAL,
            discharge_kW REAL,
            efficiency_percent REAL,
            model TEXT,
            units INTEGER
        )""",
        "transpose": False,
        "process": None
    },
    "premium_charger_edp": {
        "sheet_name": "General Information",
        "rectangle": "K17:M18",
        "df_column_names": ["type", "capacity_kW", "units"],
        "schema": """
        CREATE TABLE IF NOT EXISTS premium_charger_edp (
                    type INTEGER PRIMARY KEY,
                    capacity_kW REAL,
                    units INTEGER
        )""",
        "transpose": False,
        "process": None
    },
    "regular_ev_models": {
        "sheet_name": "General Information",
        "rectangle": "T5:AB19",
        "df_column_names" : ["model_id", "brand", "model_name", "capacity_kW", "charge_kW", "discharge_kW", "efficiency_percent", "consumption_Wh_km", "units"],
        "schema": """
        CREATE TABLE IF NOT EXISTS regular_ev_models (
            model_id INTEGER PRIMARY KEY,
            brand TEXT,
            model_name TEXT,
            capacity_kW REAL,
            charge_kW REAL,
            discharge_kW REAL,
            efficiency_percent REAL,
            consumption_Wh_km REAL,
            units INTEGER
        )""",
        "transpose": False,
        "process": None
    },
    "premium_ev_models": {
        "sheet_name": "General Information",
        "rectangle": "T24:AB33",
        "df_column_names" : ["model_id", "brand", "model_name", "capacity_kW", "charge_kW", "discharge_kW", "efficiency_percent", "consumption_Wh_km", "units"],
        "schema": """
        CREATE TABLE IF NOT EXISTS premium_ev_models (
            model_id INTEGER PRIMARY KEY,
            brand TEXT,
            model_name TEXT,
            capacity_kW REAL,
            charge_kW REAL,
            discharge_kW REAL,
            efficiency_percent REAL,
            consumption_Wh_km REAL,
            units INTEGER
        )""",
        "transpose": False,
        "process": None
    },
    "ev_departures_arrivals": {
        "sheet_name": "General Information",
        "rectangle": "AE6:AK33",
        "df_column_names": ["period", "departures_ev1", "arrivals_ev1", "departures_ev2", "arrivals_ev2", "departures_total", "arrivals_total"],
        "schema": """
        CREATE TABLE IF NOT EXISTS ev_departures_arrivals (
            period INTEGER,
            departures_ev1 INTEGER,
            arrivals_ev1 INTEGER,
            departures_ev2 INTEGER,
            arrivals_ev2 INTEGER,
            departures_total INTEGER,
            arrivals_total INTEGER
        )""",
        "transpose": False,
        "process": lambda df: df.dropna(how="all")  # drop empty rows
    },
    "mobile_ev_chargers": {
        "sheet_name": "MOBIe EV chargers",
        "rectangle": "A3:O86",
        "df_column_names": ["charger_id", "charger_parent_uid", "charger_uid", "type_of_charging", "state_of_charger","city",	"address", "operator", "capacity_kW", "voltage_level", "price_eur_per_charge",	"price_eur_per_minute",	"price_eur_per_kWh","price_eur_per_h",	"price_eur_per_kWh_2"],
        "schema": """
        CREATE TABLE IF NOT EXISTS mobile_ev_chargers (
            charger_id INTEGER PRIMARY KEY,
            charger_parent_uid INTEGER,
            charger_uid TEXT,
            type_of_charging TEXT,
            state_of_charger TEXT,
            city TEXT,
            address TEXT,
            operator TEXT,
            capacity_kW REAL,
            voltage_level TEXT,
            price_eur_per_charge REAL,
            price_eur_per_minute REAL,
            price_eur_per_kWh REAL,
            price_eur_per_h REAL,
            price_eur_per_kWh_2 REAL
        )""",
        "transpose": False,
        "process": None
    },
    "player_evs": {
        "sheet_name": "EVs",
        "rectangle": "B1:IQ3",
        "df_column_names": ["player_id", "ev1_model", "ev2_model"],
        "schema": """
        CREATE TABLE IF NOT EXISTS player_evs (
            player_id INTEGER PRIMARY KEY,
            ev1_model TEXT,
            ev2_model TEXT
        )""",
        "transpose": True,
        "process": None      # we combine this with player_devices later
    },
    "ev1_data": {
        "sheet_name": "EVs",
        "rectangle": "B6:IQ25",
        "df_column_names": [
            "player_id",
            "model_id",
            "capacity_kw",
            "charge_kw",
            "discharge_kw",
            "efficiency",
            "initial_battery_level_kw",
            "final_battery_level_kw",
            "consumption_wh_per_km",
            "departure_period",
            "arrival_period",
            "distance_km",
            "consumption_kwh",
            "morning_trip_duration_periods",
            "afternoon_trip_duration_periods",
            "trip_duration_periods",
            "charger_type",
            "public_charger",
            "power_kw",
            "price_at_public_charge_station_eur"],
        "schema": """
        CREATE TABLE IF NOT EXISTS ev1_data (
            player_id INTEGER PRIMARY KEY,
            model_id INTEGER,
            capacity_kw REAL,
            charge_kw REAL,
            discharge_kw REAL,
            efficiency REAL,
            initial_battery_level_kw REAL,
            final_battery_level_kw REAL,
            consumption_wh_per_km REAL,
            departure_period INTEGER,
            arrival_period INTEGER,
            distance_km REAL,
            consumption_kwh REAL,
            morning_trip_duration_periods INTEGER,
            afternoon_trip_duration_periods INTEGER,
            trip_duration_periods INTEGER,
            charger_type INTEGER,
            public_charger INTEGER,
            power_kw REAL,
            price_at_public_charge_station_eur REAL
        )""",
        "transpose": True,
        "process": None
    },
    "ev2_data": {
        "sheet_name": "EVs",
        "rectangle": "B27:IQ46",
        "df_column_names": [
            "player_id",
            "model_id",
            "capacity_kw",
            "charge_kw",
            "discharge_kw",
            "efficiency",
            "initial_battery_level_kw",
            "final_battery_level_kw",
            "consumption_wh_per_km",
            "departure_period",
            "arrival_period",
            "distance_km",
            "consumption_kwh",
            "morning_trip_duration_periods",
            "afternoon_trip_duration_periods",
            "trip_duration_periods",
            "charger_type",
            "public_charger",
            "power_kw",
            "price_at_public_charge_station_eur"],
        "schema": """
            CREATE TABLE IF NOT EXISTS ev2_data (
                player_id INTEGER PRIMARY KEY,
                model_id INTEGER,
                capacity_kw REAL,
                charge_kw REAL,
                discharge_kw REAL,
                efficiency REAL,
                initial_battery_level_kw REAL,
                final_battery_level_kw REAL,
                consumption_wh_per_km REAL,
                departure_period INTEGER,
                arrival_period INTEGER,
                distance_km REAL,
                consumption_kwh REAL,
                morning_trip_duration_periods INTEGER,
                afternoon_trip_duration_periods INTEGER,
                trip_duration_periods INTEGER,
                charger_type INTEGER,
                public_charger INTEGER,
                power_kw REAL,
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
        "df_column_names": ["player_id", "model_id", "capacity_kW", "charge_kW", "discharge_kW", "efficiency", "initial_kW", "final_kW"],
        "schema": """
        CREATE TABLE IF NOT EXISTS bess (
            player_id INTEGER PRIMARY KEY,
            model_id INTEGER,
            capacity_kW REAL,
            charge_kW REAL,
            discharge_kW REAL,
            efficiency REAL,
            initial_kW REAL,
            final_kW REAL
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
        "df_column_names": ["player_id", "power_buy_kW", "power_sell_kW", "fixed_costs_eur","_","initial_cp_kW","premium_charger_edp_capacity_kW","sum_kW","new_cp_level_kW","fixed_costs_2_eur"],
        "schema": """
        CREATE TABLE IF NOT EXISTS limits (
            player_id INTEGER PRIMARY KEY,
            power_buy_kW REAL,
            power_sell_kW REAL,
            fixed_costs_eur REAL,
            initial_cp_kW REAL,
            premium_charger_edp_capacity_kW REAL,
            sum_kW REAL,
            new_cp_level_kW REAL,
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
    }
}