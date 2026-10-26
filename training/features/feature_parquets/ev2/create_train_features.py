import random
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = ROOT_DIR / "sqlite" / "energy_v2_training.db"
PERIODS_PER_DAY = 96
PROFILES_PER_CHUNK = 100
MAX_PROFILES = 1_000
RANDOM_SEED = 42
TARGET_OFFSETS = (2, 4, 8, 16, 32, 64)
CHUNK_NAME_PATTERN = re.compile(r"ev2_status_features_\d+-\d+\.parquet")

sys.path.insert(0, str(ROOT_DIR))

from training.features.ev_status_features import _add_ev_status_features  # noqa: E402


def _get_profile_ids(connection: sqlite3.Connection) -> list[str]:
    profile_ids = [
        row[1]
        for row in connection.execute('PRAGMA table_info("ev2_status")').fetchall()
        if row[1] != "period"
    ]
    invalid_ids = [profile_id for profile_id in profile_ids if not re.fullmatch(r"\d+", profile_id)]
    if invalid_ids:
        raise ValueError(f"Unexpected EV2 profile identifiers: {invalid_ids}")
    if not profile_ids:
        raise ValueError("The ev2_status table contains no profiles")
    return profile_ids


def _get_person_metadata(connection: sqlite3.Connection) -> dict[str, dict[str, str]]:
    rows = connection.execute(
        "SELECT player_id, persid, HHID FROM ev2_persons ORDER BY player_id"
    ).fetchall()
    metadata = {
        str(player_id): {"profile_id": str(persid), "household_id": str(hhid)}
        for player_id, persid, hhid in rows
    }
    return metadata


def _read_profiles(
    connection: sqlite3.Connection,
    profile_ids: list[str],
) -> pd.DataFrame:
    selected_profiles = ", ".join(f'"{profile_id}"' for profile_id in profile_ids)
    profiles = pd.read_sql_query(
        f'SELECT period, {selected_profiles} FROM ev2_status ORDER BY period',
        connection,
    )
    if profiles["period"].tolist() != list(range(1, PERIODS_PER_DAY + 1)):
        raise ValueError(f"Expected EV2 periods 1-{PERIODS_PER_DAY}")
    if profiles[profile_ids].isna().any().any():
        raise ValueError("Selected EV2 profiles contain missing statuses")
    statuses = set(pd.unique(profiles[profile_ids].to_numpy().ravel()))
    if not statuses.issubset({0, 1, 2}):
        raise ValueError(f"Unexpected EV2 status values: {sorted(statuses)}")
    return profiles


def _make_profile_features(
    profiles: pd.DataFrame,
    player_id: str,
    metadata: dict[str, str],
) -> pd.DataFrame:
    feature_df = pd.DataFrame(
        {
            "profile_id": metadata["profile_id"],
            "household_id": metadata["household_id"],
            "player_id": int(player_id),
            "ev_key": "ev2",
            "timestep": profiles["period"].astype(int),
            "status": profiles[player_id].astype(int),
        }
    )
    feature_df = _add_ev_status_features(
        feature_df,
        group_cols=("profile_id",),
        target_fill_value=-1,
    )
    feature_df["y1"] = feature_df["next_state"]
    grouped_status = feature_df.groupby("profile_id")["status"]
    for offset in TARGET_OFFSETS:
        feature_df[f"y{offset}"] = grouped_status.shift(-offset).fillna(-1).astype(int)

    leading_columns = ["profile_id", "household_id", "timestep", "status"]
    feature_df = feature_df[leading_columns + [
        column for column in feature_df.columns if column not in leading_columns
    ]]
    float_columns = feature_df.select_dtypes(include="floating").columns
    feature_df[float_columns] = feature_df[float_columns].round(3)
    return feature_df


def _write_chunk(
    profiles: list[pd.DataFrame],
    first_profile: int,
) -> None:
    last_profile = first_profile + len(profiles) - 1
    output_path = OUTPUT_DIR / f"ev2_status_features_{first_profile}-{last_profile}.parquet"
    chunk = pd.concat(profiles, ignore_index=True)
    chunk.to_parquet(output_path, index=False)
    print(f"Wrote {output_path.name}: {len(profiles)} profiles, {len(chunk)} rows")


def create_train_features() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        profile_ids = _get_profile_ids(connection)
        metadata = _get_person_metadata(connection)
        missing_metadata = sorted(set(profile_ids) - metadata.keys())
        if missing_metadata:
            raise ValueError(f"Missing EV2 metadata for player IDs: {missing_metadata}")
        profiles = _read_profiles(connection, profile_ids)

    profile_count = min(MAX_PROFILES, len(profile_ids))
    selected_profiles = random.Random(RANDOM_SEED).sample(profile_ids, profile_count)
    profile_features = [
        _make_profile_features(
            profiles,
            player_id,
            metadata[player_id],
        )
        for player_id in selected_profiles
    ]

    for start in range(0, profile_count, PROFILES_PER_CHUNK):
        _write_chunk(
            profile_features[start : start + PROFILES_PER_CHUNK],
            start + 1,
        )

    for output_path in OUTPUT_DIR.iterdir():
        if output_path.is_file() and CHUNK_NAME_PATTERN.fullmatch(output_path.name):
            match = re.fullmatch(r"ev2_status_features_(\d+)-(\d+)\.parquet", output_path.name)
            if match and int(match.group(2)) > profile_count:
                output_path.unlink()


if __name__ == "__main__":
    create_train_features()
