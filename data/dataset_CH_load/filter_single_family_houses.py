'''
This script filters the smart meter data to only include single family houses (installation type)
that dont have additional devices.

This way we hope to get data comparable to our original dataset to use for base load training.

filters:
1_ewh, 1_hp, 1_hp-add, 1_hp-wh, 1_ev, 1_storage_heating, 1_direct_heating, 1_change = False
0_installation_type = Single-family house

'''

import shutil
from pathlib import Path

import pandas as pd

DATASET_DIR = Path(__file__).parent
METADATA_PATH = DATASET_DIR / 'metadata.csv'
SMART_METER_DATA_DIR = DATASET_DIR / 'smart_meter_data'
FILTERED_METADATA_PATH = DATASET_DIR / 'metadata_filtered.csv'
FILTERED_SMART_METER_DATA_DIR = DATASET_DIR / 'smart_meter_data_filtered'


def filter_single_family_houses(raw_metadata: pd.DataFrame) -> pd.DataFrame:
    """Filter metadata for single-family houses without additional devices."""
    single_family_houses = raw_metadata[raw_metadata['0_installation_type'] == 'Single-family house']
    filtered_sfh_metadata = single_family_houses[
                                    (~single_family_houses['1_ewh']) &
                                    (~single_family_houses['1_hp']) &
                                    (~single_family_houses['1_hp-add']) &
                                    (~single_family_houses['1_hp-wh']) &
                                    (~single_family_houses['1_ev']) &
                                    (~single_family_houses['1_storage_heating']) &
                                    (~single_family_houses['1_direct_heating']) &
                                    (~single_family_houses['1_change']) &
                                    (single_family_houses['0_num_data_points'] >= 70000)
                                ]

    return filtered_sfh_metadata


def write_filtered_dataset(filtered_metadata: pd.DataFrame) -> None:
    """Write the filtered metadata csv and copy the matching smart meter csvs."""
    filtered_metadata.to_csv(FILTERED_METADATA_PATH, sep=';')

    FILTERED_SMART_METER_DATA_DIR.mkdir(exist_ok=True)
    missing_files = []
    for meter_id in filtered_metadata.index:
        src = SMART_METER_DATA_DIR / f"{meter_id}.csv"
        if not src.exists():
            missing_files.append(meter_id)
            continue
        shutil.copy2(src, FILTERED_SMART_METER_DATA_DIR / src.name)

    if missing_files:
        print(f"Warning: {len(missing_files)} smart meter csv files were missing: {missing_files}")


def main():
    raw_metadata = pd.read_csv(METADATA_PATH, sep=';', index_col=0)
    filtered_sfh_metadata = filter_single_family_houses(raw_metadata)
    print(f"Filtered metadata contains {len(filtered_sfh_metadata)} single family houses without additional devices. (70000+ datapoints each)")
    write_filtered_dataset(filtered_sfh_metadata)
    print(f"Wrote {FILTERED_METADATA_PATH.name} and copied csvs to {FILTERED_SMART_METER_DATA_DIR.name}/")


if __name__ == "__main__":
    main()