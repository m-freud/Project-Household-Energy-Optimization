import random
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = ROOT_DIR / "sqlite" / "source_runtime_training.db"
FIRST_DATE = "2023-01-01"
PERIODS_PER_DAY = 96
PROFILES_PER_CHUNK = 100
HOUSEHOLDS_PER_HOLDOUT_SPLIT = 5
PROFILES_PER_SPLIT = {"train": 800, "test": 100, "validation": 100}
RANDOM_SEED = 42
TARGET_OFFSETS = tuple(range(1, PERIODS_PER_DAY))
CHUNK_NAME_PATTERN = re.compile(r"ch_load_features_\d+-\d+\.parquet")

sys.path.insert(0, str(ROOT_DIR))

from training.features.base_load_features import _add_base_load_features  # noqa: E402


def _get_household_columns(connection: sqlite3.Connection) -> list[str]:
    columns = [
        row[1]
        for row in connection.execute('PRAGMA table_info("load")').fetchall()
        if row[1] not in {"timestamp_utc", "period"}
    ]
    invalid_columns = [column for column in columns if not re.fullmatch(r"\d+", column)]
    if invalid_columns:
        raise ValueError(f"Unexpected household columns in load table: {invalid_columns}")
    if not columns:
        raise ValueError("The load table contains no household columns")
    return columns


def _get_complete_dates(connection: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            """
            SELECT substr(timestamp_utc, 1, 10)
            FROM load
            WHERE substr(timestamp_utc, 1, 10) >= ?
            GROUP BY substr(timestamp_utc, 1, 10)
            ORDER BY substr(timestamp_utc, 1, 10)
            """,
            (FIRST_DATE,),
        ).fetchall()
    ]


def _read_complete_day(
    connection: sqlite3.Connection,
    date: str,
    household_columns: list[str],
) -> pd.DataFrame | None:
    selected_households = ", ".join(f'"{column}"' for column in household_columns)
    day = pd.read_sql_query(
        f"""
        SELECT period, {selected_households}
        FROM load
        WHERE substr(timestamp_utc, 1, 10) = ?
        ORDER BY period
        """,
        connection,
        params=(date,),
    )
    if day["period"].tolist() != list(range(1, PERIODS_PER_DAY + 1)):
        print(f"Skipping {date}: expected exactly periods 1-{PERIODS_PER_DAY}")
        return None
    return day


def _get_available_profiles(
    connection: sqlite3.Connection,
    dates: list[str],
    household_columns: list[str],
) -> list[tuple[str, str]]:
    available_profiles: list[tuple[str, str]] = []
    for date in dates:
        day = _read_complete_day(connection, date, household_columns)
        if day is None:
            continue
        available_profiles.extend(
            (household_id, date)
            for household_id in household_columns
            if not day[household_id].isna().any()
        )
    return available_profiles


def _split_households(household_columns: list[str]) -> dict[str, list[str]]:
    holdout_household_count = 2 * HOUSEHOLDS_PER_HOLDOUT_SPLIT
    if len(household_columns) <= holdout_household_count:
        raise ValueError(
            f"Need at least {holdout_household_count + 1} households to create "
            "disjoint train, test, and validation splits"
        )

    selected_holdout_households = random.Random(RANDOM_SEED).sample(
        household_columns,
        holdout_household_count,
    )
    test_households = selected_holdout_households[:HOUSEHOLDS_PER_HOLDOUT_SPLIT]
    validation_households = selected_holdout_households[HOUSEHOLDS_PER_HOLDOUT_SPLIT:]
    holdout_households = set(selected_holdout_households)
    train_households = [
        household_id
        for household_id in household_columns
        if household_id not in holdout_households
    ]
    return {
        "train": train_households,
        "test": test_households,
        "validation": validation_households,
    }


def _sample_profiles_by_split(
    available_profiles: list[tuple[str, str]],
    split_households: dict[str, list[str]],
) -> dict[str, list[tuple[str, str]]]:
    profiles_by_split: dict[str, list[tuple[str, str]]] = {}
    for seed_offset, (split, household_ids) in enumerate(split_households.items()):
        allowed_households = set(household_ids)
        split_profiles = [
            profile
            for profile in available_profiles
            if profile[0] in allowed_households
        ]
        profile_count = PROFILES_PER_SPLIT[split]
        if len(split_profiles) < profile_count:
            raise ValueError(
                f"Found only {len(split_profiles)} complete household-day profiles "
                f"for {split}; need {profile_count}"
            )
        profiles_by_split[split] = random.Random(RANDOM_SEED + seed_offset).sample(
            split_profiles,
            profile_count,
        )
    return profiles_by_split


def _make_daily_features(
    day: pd.DataFrame,
    date: str,
    household_columns: list[str],
) -> pd.DataFrame:
    feature_df = day[["period", *household_columns]].melt(
        id_vars=["period"],
        var_name="household_id",
        value_name="base_load",
    )
    feature_df = feature_df.rename(columns={"period": "timestep"})
    feature_df["household_id"] = feature_df["household_id"].astype(int)
    feature_df["date"] = date
    feature_df["profile_id"] = feature_df["household_id"].astype(str) + "-" + date
    feature_df["base_load"] = feature_df["base_load"].astype(float)
    feature_df = feature_df.sort_values(
        ["household_id", "timestep"]
    ).reset_index(drop=True)

    group_cols = ("household_id", "date")
    feature_df = _add_base_load_features(
        feature_df,
        group_cols=group_cols,
        target_fill_value=-1.0,
    )
    grouped_values = feature_df.groupby(list(group_cols))["base_load"]
    for offset in TARGET_OFFSETS:
        feature_df[f"y{offset}"] = grouped_values.shift(-offset).fillna(-1.0).astype(float)
    leading_columns = ["profile_id", "household_id", "date", "timestep", "base_load"]
    feature_df = feature_df[leading_columns + [
        column for column in feature_df.columns if column not in leading_columns
    ]]
    float_columns = feature_df.select_dtypes(include="floating").columns
    feature_df[float_columns] = feature_df[float_columns].round(3)
    return feature_df


def _write_chunk(
    profiles: list[pd.DataFrame],
    first_profile: int,
    output_dir: Path,
) -> int:
    last_profile = first_profile + len(profiles) - 1
    output_path = output_dir / f"ch_load_features_{first_profile}-{last_profile}.parquet"
    chunk = pd.concat(profiles, ignore_index=True)
    chunk.to_parquet(output_path, index=False)
    print(f"Wrote {output_path.name}: {len(profiles)} profiles, {len(chunk)} rows")
    return last_profile + 1


def create_train_features() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        household_columns = _get_household_columns(connection)
        dates = _get_complete_dates(connection)
        available_profiles = _get_available_profiles(connection, dates, household_columns)
        split_households = _split_households(household_columns)
        selected_by_split = _sample_profiles_by_split(
            available_profiles,
            split_households,
        )

        for split, selected_profiles in selected_by_split.items():
            selected_by_date: dict[str, list[str]] = {}
            for household_id, date in selected_profiles:
                selected_by_date.setdefault(date, []).append(household_id)

            feature_by_profile: dict[tuple[str, str], pd.DataFrame] = {}
            for date, selected_households in selected_by_date.items():
                day = _read_complete_day(connection, date, selected_households)
                if day is None or day[selected_households].isna().any().any():
                    raise ValueError(
                        f"Selected {split} household-day profiles are incomplete "
                        f"for {date}"
                    )
                daily_features = _make_daily_features(day, date, selected_households)
                for household_id, profile in daily_features.groupby(
                    "household_id",
                    sort=False,
                ):
                    feature_by_profile[(str(household_id), date)] = profile.reset_index(
                        drop=True
                    )

            split_output_dir = OUTPUT_DIR / split
            split_output_dir.mkdir(parents=True, exist_ok=True)
            ordered_profiles = [
                feature_by_profile[profile] for profile in selected_profiles
            ]
            profile_count = PROFILES_PER_SPLIT[split]
            for start in range(0, profile_count, PROFILES_PER_CHUNK):
                _write_chunk(
                    ordered_profiles[start : start + PROFILES_PER_CHUNK],
                    start + 1,
                    split_output_dir,
                )

            for output_path in split_output_dir.iterdir():
                if (
                    output_path.is_file()
                    and CHUNK_NAME_PATTERN.fullmatch(output_path.name)
                ):
                    match = re.fullmatch(
                        r"ch_load_features_(\d+)-(\d+)\.parquet",
                        output_path.name,
                    )
                    if match and int(match.group(2)) > profile_count:
                        output_path.unlink()

            print(
                f"{split}: {len(selected_profiles)} profiles, "
                f"{len(split_households[split])} households"
            )


if __name__ == "__main__":
    create_train_features()
