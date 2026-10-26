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
CHUNK_NAME_PATTERN = re.compile(r"de_pv_features_\d+-\d+\.parquet")

sys.path.insert(0, str(ROOT_DIR))

from training.features.pv_gen_features import _add_pv_gen_features  # noqa: E402


def _get_player_columns(connection: sqlite3.Connection) -> list[str]:
    columns = [
        row[1]
        for row in connection.execute('PRAGMA table_info("pv_gen")').fetchall()
        if row[1] not in {"timestamp_utc", "period"}
    ]
    invalid_columns = [column for column in columns if not re.fullmatch(r"[A-Za-z0-9_]+", column)]
    if invalid_columns:
        raise ValueError(f"Unexpected player columns in pv_gen table: {invalid_columns}")
    if not columns:
        raise ValueError("The pv_gen table contains no PV player columns")
    return columns


def _get_dates(connection: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            """
            SELECT substr(timestamp_utc, 1, 10)
            FROM pv_gen
            GROUP BY substr(timestamp_utc, 1, 10)
            ORDER BY substr(timestamp_utc, 1, 10)
            """
        ).fetchall()
    ]


def _read_day(
    connection: sqlite3.Connection,
    date: str,
    player_ids: list[str],
) -> pd.DataFrame | None:
    selected_players = ", ".join(f'"{player_id}"' for player_id in player_ids)
    day = pd.read_sql_query(
        f"""
        SELECT period, {selected_players}
        FROM pv_gen
        WHERE substr(timestamp_utc, 1, 10) = ?
        ORDER BY period
        """,
        connection,
        params=(date,),
    )
    if day["period"].tolist() != list(range(1, PERIODS_PER_DAY + 1)):
        return None
    return day


def _get_available_profiles(
    connection: sqlite3.Connection,
    dates: list[str],
    player_columns: list[str],
) -> list[tuple[str, str]]:
    available_profiles: list[tuple[str, str]] = []
    for date in dates:
        day = _read_day(connection, date, player_columns)
        if day is None:
            continue
        available_profiles.extend(
            (player_id, date)
            for player_id in player_columns
            if not day[player_id].isna().any()
        )
    return available_profiles


def _make_daily_features(
    day: pd.DataFrame,
    date: str,
    player_ids: list[str],
) -> pd.DataFrame:
    feature_df = day[["period", *player_ids]].melt(
        id_vars=["period"],
        var_name="player_id",
        value_name="pv_gen",
    )
    feature_df = feature_df.rename(columns={"period": "timestep"})
    feature_df["date"] = date
    feature_df["profile_id"] = feature_df["player_id"] + "-" + date
    feature_df["pv_gen"] = feature_df["pv_gen"].astype(float)
    feature_df = feature_df.sort_values(
        ["player_id", "timestep"]
    ).reset_index(drop=True)

    group_cols = ("player_id", "date")
    feature_df = _add_pv_gen_features(
        feature_df,
        group_cols=group_cols,
        target_fill_value=-1.0,
    )
    feature_df["y1"] = feature_df["next_value"]
    grouped_values = feature_df.groupby(list(group_cols))["pv_gen"]
    for offset in TARGET_OFFSETS:
        feature_df[f"y{offset}"] = grouped_values.shift(-offset).fillna(-1.0).astype(float)

    leading_columns = ["profile_id", "player_id", "date", "timestep", "pv_gen"]
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
) -> None:
    last_profile = first_profile + len(profiles) - 1
    output_path = output_dir / f"de_pv_features_{first_profile}-{last_profile}.parquet"
    chunk = pd.concat(profiles, ignore_index=True)
    chunk.to_parquet(output_path, index=False)
    print(f"Wrote {output_path.name}: {len(profiles)} profiles, {len(chunk)} rows")


def create_train_features() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        player_columns = _get_player_columns(connection)
        dates = _get_dates(connection)
        available_profiles = _get_available_profiles(connection, dates, player_columns)
        profile_count = min(MAX_PROFILES, len(available_profiles))
        if profile_count == 0:
            raise ValueError("No complete pool-filtered PV profile-days are available")

        selected_profiles = random.Random(RANDOM_SEED).sample(
            available_profiles,
            profile_count,
        )
        selected_by_date: dict[str, list[str]] = {}
        for player_id, date in selected_profiles:
            selected_by_date.setdefault(date, []).append(player_id)

        feature_by_profile: dict[tuple[str, str], pd.DataFrame] = {}
        for date, selected_players in selected_by_date.items():
            day = _read_day(connection, date, selected_players)
            if day is None or day[selected_players].isna().any().any():
                raise ValueError(f"Selected PV profiles are incomplete for {date}")
            daily_features = _make_daily_features(day, date, selected_players)
            for player_id, profile in daily_features.groupby("player_id", sort=False):
                feature_by_profile[(player_id, date)] = profile.reset_index(drop=True)

    ordered_profiles = [feature_by_profile[profile] for profile in selected_profiles]
    for start in range(0, profile_count, PROFILES_PER_CHUNK):
        _write_chunk(
            ordered_profiles[start : start + PROFILES_PER_CHUNK],
            start + 1,
            OUTPUT_DIR,
        )

    for output_path in OUTPUT_DIR.iterdir():
        if output_path.is_file() and CHUNK_NAME_PATTERN.fullmatch(output_path.name):
            match = re.fullmatch(r"de_pv_features_(\d+)-(\d+)\.parquet", output_path.name)
            if match and int(match.group(2)) > profile_count:
                output_path.unlink()


if __name__ == "__main__":
    create_train_features()