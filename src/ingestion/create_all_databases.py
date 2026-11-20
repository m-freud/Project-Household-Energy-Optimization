from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
CH_DIR = ROOT_DIR / "data" / "dataset_CH_load"
RAW_METER_DIR = CH_DIR / "smart_meter_data"
FILTERED_METER_DIR = CH_DIR / "smart_meter_data_filtered"
PV_DIR = ROOT_DIR / "data" / "dataset_DE_PV"
EV_DIR = ROOT_DIR / "data" / "dataset_VISTA_EV"
STAGING_DIR = ROOT_DIR / "sqlite" / "staging"

PV_COLUMNS = (
    "DE_KN_residential1_pv",
    "DE_KN_residential3_pv",
    "DE_KN_residential4_pv",
    "DE_KN_residential6_pv",
)

STEPS: tuple[tuple[str, Path | None], ...] = (
    (
        "Load original Excel workbook",
        Path("src/ingestion/original_data/table_loading_original.py")),
    (
        "Calculate original profile metadata",
        Path("src/ingestion/original_data/create_original_metadata.py")),
    # (
    #     "Filter Swiss household measurements",
    #     Path("data/dataset_CH_load/filter_single_family_houses.py"),
    # ),
    (
        "Build Swiss load staging database",
        Path("src/ingestion/external_data/load/create_ch_database.py"),
    ),
    (
        "Calculate Swiss load metadata",
        Path("src/ingestion/external_data/load/create_metadata_table.py"),
    ),
    (
        "Build Swiss load candidate pool",
        Path("src/ingestion/external_data/load/create_valid_pool_table.py"),
    ),
    (
        "Select Swiss runtime load profiles",
        Path("src/ingestion/external_data/load/select_ext_runtime_profiles.py"),
    ),
    (
        "Prepare PV input from raw CSV",
        None
    ),
    (
        "Build German PV staging database",
        Path("src/ingestion/external_data/pv/create_pv_database.py"),
    ),
    (
        "Calculate German PV metadata",
        Path("src/ingestion/external_data/pv/create_metadata_table.py"),
    ),
    (
        "Build German PV candidate pool",
        Path("src/ingestion/external_data/pv/create_valid_pool_table.py"),
    ),
    (
        "Select German PV runtime pool",
        Path("src/ingestion/external_data/pv/select_ext_runtime_profiles.py"),
    ),
    (
        "Filter VISTA EV1 commute pairs",
        Path("data/dataset_VISTA_EV/create_work_commute_pairs_table.py"),
    ),
    (
        "Build VISTA EV1 staging database",
        Path("src/ingestion/external_data/ev1/create_ev1_database.py"),
    ),
    (
        "Calculate VISTA EV1 metadata",
        Path("src/ingestion/external_data/ev1/create_ev1_metadata_table.py"),
    ),
    (
        "Filter VISTA EV2 casual commute chains",
        Path("data/dataset_VISTA_EV/create_casual_commutes_table.py"),
    ),
    (
        "Build VISTA EV2 staging database",
        Path("src/ingestion/external_data/ev2/create_ev2_database.py"),
    ),
    # (
    #     "Calculate VISTA EV2 metadata",
    #     Path("data/dataset_VISTA_EV/create_ev2_metadata_table.py"),
    # ),
    ("Build runtime database", Path("src/ingestion/external_data/create_replacement_db.py")),
    ("Build training database", Path("src/ingestion/external_data/create_training_tables.py")),
)


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def _required_inputs() -> list[Path]:
    excel_path = Path(
        os.environ.get(
            "EXCEL_FILE_PATH",
            str(ROOT_DIR / "data" / "dataset_original" / "EC_EV_dataset.xlsx"),
        )
    )
    inputs = [
        _resolve_path(excel_path),
        EV_DIR / "T_VISTA1218_V1.csv",
        PV_DIR / "pv_data_raw.csv",
    ]
    if _has_csv_files(RAW_METER_DIR):
        inputs.append(CH_DIR / "metadata.csv")
    return inputs


def _has_csv_files(directory: Path) -> bool:
    return directory.is_dir() and any(directory.glob("*.csv"))


def _use_filtered_load_data() -> bool:
    return not _has_csv_files(RAW_METER_DIR) and _has_csv_files(FILTERED_METER_DIR)


def _missing_inputs() -> list[str]:
    missing = [str(path) for path in _required_inputs() if not path.is_file()]
    if not _has_csv_files(RAW_METER_DIR) and not _has_csv_files(FILTERED_METER_DIR):
        missing.append(
            f"{RAW_METER_DIR} or {FILTERED_METER_DIR} "
            "(expected household CSV files)"
        )
    return missing


def _write_non_empty_pv_csv(raw_path: Path, output_path: Path) -> int:
    temp_path: Path | None = None
    try:
        with raw_path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise ValueError(f"PV source has no header: {raw_path}")
            missing_columns = set(PV_COLUMNS) - set(reader.fieldnames)
            if missing_columns:
                raise ValueError(
                    f"PV source is missing generation columns: {sorted(missing_columns)}"
                )

            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                dir=output_path.parent,
                suffix=".tmp",
                delete=False,
            ) as target:
                temp_path = Path(target.name)
                writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
                writer.writeheader()
                retained = 0
                for row in reader:
                    if any((row.get(column) or "").strip() for column in PV_COLUMNS):
                        for column in PV_COLUMNS:
                            value = (row.get(column) or "").strip()
                            if value:
                                try:
                                    row[column] = str(float(value))
                                except ValueError as error:
                                    raise ValueError(
                                        f"Invalid PV value {value!r} in {column}"
                                    ) from error
                        writer.writerow(row)
                        retained += 1

        if retained == 0:
            raise ValueError(f"No rows with PV generation found in {raw_path}")
        if temp_path is None:
            raise RuntimeError("PV output temporary file was not created")
        temp_path.replace(output_path)
        return retained
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _verify_filtered_load_inputs() -> None:
    metadata_path = CH_DIR / "metadata_filtered.csv"
    available_ids = {path.stem for path in FILTERED_METER_DIR.glob("*.csv")}
    if metadata_path.is_file():
        with metadata_path.open("r", newline="", encoding="utf-8-sig") as source:
            rows = csv.reader(source, delimiter=";")
            next(rows, None)
            selected_ids = {row[0].strip() for row in rows if row and row[0].strip()}
    else:
        selected_ids = available_ids

    missing_ids = sorted(selected_ids - available_ids)
    if missing_ids:
        raise RuntimeError(
            "Filtered CH metadata has households without filtered measurement "
            "files: "
            + ", ".join(missing_ids)
        )
    if not available_ids:
        raise RuntimeError(f"No filtered CH household CSV files found in {FILTERED_METER_DIR}")


def _run_step(label: str, script: Path) -> None:
    command = [sys.executable, str(ROOT_DIR / script)]
    print(f"\n== {label} ==")
    environment = os.environ.copy()
    environment["INIT_SOURCE_SQLITE_PATH"] = str(
        ROOT_DIR / "sqlite" / "source_original.db"
    )
    subprocess.run(command, cwd=ROOT_DIR, check=True, env=environment)


def _verify_database(path: Path, required_tables: set[str]) -> None:
    if not path.is_file():
        raise RuntimeError(f"Expected database was not created: {path}")
    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    missing_tables = required_tables - tables
    if missing_tables:
        raise RuntimeError(
            f"{path} is missing tables: {', '.join(sorted(missing_tables))}"
        )


def _verify_outputs() -> None:
    databases = {
        ROOT_DIR / "sqlite" / "source_original.db": {
            "base_load",
            "base_load_metadata",
            "pv_gen",
            "pv_gen_metadata",
            "initial_cp",
            "ev1_status",
            "ev1_metadata",
            "ev2_status",
            "ev2_metadata",
        },
        STAGING_DIR / "CH_loads.db": {
            "load",
            "base_load_metadata",
            "base_load_pool",
            "runtime_pool_by_cp",
        },
        STAGING_DIR / "DE_pv_gen.db": {
            "pv_gen",
            "pv_gen_metadata",
            "pv_gen_pool",
            "runtime_pool",
        },
        STAGING_DIR / "VISTA_ev1_status.db": {
            "ev1_status",
            "ev1_distances",
            "ev1_persons",
            "ev1_metadata",
        },
        STAGING_DIR / "VISTA_ev2_status.db": {
            "ev2_status",
            "ev2_distances",
            "ev2_persons",
            "ev2_metadata",
        },
        ROOT_DIR / "sqlite" / "source_runtime.db": {
            "base_load",
            "pv_gen",
            "ev1_status",
            "ev2_status",
            "ev1_load",
            "ev2_load",
            "ev1_at_home",
            "ev2_at_home",
            "ev1_at_charging_station",
            "ev2_at_charging_station",
            "ev1_max_charge",
            "ev2_max_charge",
            "ev1_buy_price",
            "ev2_buy_price",
            "ev2_persons",
        },
        ROOT_DIR / "sqlite" / "source_runtime_training.db": {
            "load",
            "pv_gen",
            "ev1_status",
            "ev2_status",
            "ev2_persons",
        },
    }
    for path, required_tables in databases.items():
        _verify_database(path, required_tables)


def run_pipeline() -> None:
    missing = _missing_inputs()
    if missing:
        raise FileNotFoundError(
            "Cannot build databases; required starting inputs are missing:\n- "
            + "\n- ".join(missing)
        )

    use_filtered_load_data = _use_filtered_load_data()
    if use_filtered_load_data:
        print(
            "Raw CH measurements not found; using existing filtered files in "
            f"{FILTERED_METER_DIR}"
        )

    for label, script in STEPS:
        if (
            script == Path("data/dataset_CH_load/filter_single_family_houses.py")
            and use_filtered_load_data
        ):
            _verify_filtered_load_inputs()
            print(f"\n== Using existing filtered CH data: {FILTERED_METER_DIR} ==")
            continue
        if script is None:
            pv_rows = _write_non_empty_pv_csv(
                PV_DIR / "pv_data_raw.csv",
                PV_DIR / "pv_data_non_empty.csv",
            )
            print(
                f"Prepared PV input with {pv_rows} rows: "
                f"{PV_DIR / 'pv_data_non_empty.csv'}"
            )
        else:
            _run_step(label, script)
            if script == Path("data/dataset_CH_load/filter_single_family_houses.py"):
                _verify_filtered_load_inputs()

    _verify_outputs()
    print("\nAll source, staging, runtime, and training databases were created.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the original, staging, runtime, and training databases."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the build order and report missing starting inputs without writing files.",
    )
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    if args.dry_run:
        use_filtered_load_data = _use_filtered_load_data()
        for index, (label, script) in enumerate(STEPS, start=1):
            if (
                script == Path("data/dataset_CH_load/filter_single_family_houses.py")
                and use_filtered_load_data
            ):
                print(
                    f"{index:2}. Use existing filtered CH CSVs: "
                    f"{FILTERED_METER_DIR}"
                )
                continue
            suffix = (
                " (pv_data_raw.csv -> pv_data_non_empty.csv)"
                if script is None
                else f": {script}"
            )
            print(f"{index:2}. {label}{suffix}")
        missing = _missing_inputs()
        if missing:
            print("Missing starting inputs:\n- " + "\n- ".join(missing))
            raise SystemExit(1)
        if use_filtered_load_data:
            try:
                _verify_filtered_load_inputs()
            except RuntimeError as error:
                print(f"Invalid filtered CH input: {error}")
                raise SystemExit(1) from error
        print("All required starting inputs are present.")
        return

    try:
        run_pipeline()
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Database build failed: {error}") from error


if __name__ == "__main__":
    main()
