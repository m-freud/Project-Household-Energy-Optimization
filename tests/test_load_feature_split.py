import sqlite3

import pandas as pd
import pytest

from models.load import train_models
from training.features.feature_parquets.load import create_train_features


def test_household_splits_are_deterministic_and_disjoint():
    households = [str(household_id) for household_id in range(41)]

    first_split = create_train_features._split_households(households)
    second_split = create_train_features._split_households(households)

    assert first_split == second_split
    assert len(first_split["test"]) == 5
    assert len(first_split["validation"]) == 5
    assert len(first_split["train"]) == 31
    assert set(first_split["train"]).isdisjoint(first_split["test"])
    assert set(first_split["train"]).isdisjoint(first_split["validation"])
    assert set(first_split["test"]).isdisjoint(first_split["validation"])


def test_profile_sampling_is_deterministic_and_respects_household_split(monkeypatch):
    split_households = {
        "train": ["train-household"],
        "test": ["test-household"],
        "validation": ["validation-household"],
    }
    profiles = [
        (household_id, f"2023-01-{day:02d}")
        for household_id in (
            "train-household",
            "test-household",
            "validation-household",
        )
        for day in range(1, 21)
    ]
    monkeypatch.setattr(
        create_train_features,
        "PROFILES_PER_SPLIT",
        {"train": 10, "test": 5, "validation": 5},
    )

    first_sample = create_train_features._sample_profiles_by_split(
        profiles,
        split_households,
    )
    second_sample = create_train_features._sample_profiles_by_split(
        profiles,
        split_households,
    )

    assert first_sample == second_sample
    assert {split: len(selected) for split, selected in first_sample.items()} == {
        "train": 10,
        "test": 5,
        "validation": 5,
    }
    for split, selected_profiles in first_sample.items():
        assert {household_id for household_id, _ in selected_profiles} == set(
            split_households[split]
        )


def test_profile_sampling_fails_if_a_split_has_too_few_profiles(monkeypatch):
    monkeypatch.setattr(
        create_train_features,
        "PROFILES_PER_SPLIT",
        {"train": 2, "test": 2, "validation": 2},
    )

    with pytest.raises(ValueError, match="only 0 complete household-day profiles for train; need 2"):
        create_train_features._sample_profiles_by_split(
            [("test-household", "2023-01-01")],
            {
                "train": ["train-household"],
                "test": ["test-household"],
                "validation": ["validation-household"],
            },
        )


def test_create_train_features_writes_disjoint_split_directories(
    monkeypatch,
    tmp_path,
):
    database_path = tmp_path / "source.db"
    output_dir = tmp_path / "features"
    household_columns = [str(household_id) for household_id in range(1000, 1011)]
    quoted_households = ", ".join(
        f'"{household_id}" REAL' for household_id in household_columns
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            f"CREATE TABLE load (timestamp_utc TEXT, period INTEGER, {quoted_households})"
        )
        placeholders = ", ".join("?" for _ in range(2 + len(household_columns)))
        connection.executemany(
            f"INSERT INTO load VALUES ({placeholders})",
            [
                ("2023-01-01", period, *([float(period)] * len(household_columns)))
                for period in range(1, create_train_features.PERIODS_PER_DAY + 1)
            ],
        )

    monkeypatch.setattr(create_train_features, "DATABASE_PATH", database_path)
    monkeypatch.setattr(create_train_features, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(
        create_train_features,
        "PROFILES_PER_SPLIT",
        {"train": 1, "test": 1, "validation": 1},
    )

    create_train_features.create_train_features()

    split_households = {}
    for split in ("train", "test", "validation"):
        parquet_path = output_dir / split / "ch_load_features_1-1.parquet"
        assert parquet_path.is_file()
        profile = pd.read_parquet(parquet_path)
        assert len(profile) == create_train_features.PERIODS_PER_DAY
        assert profile["profile_id"].nunique() == 1
        split_households[split] = set(profile["household_id"])

    assert split_households["train"].isdisjoint(split_households["test"])
    assert split_households["train"].isdisjoint(split_households["validation"])
    assert split_households["test"].isdisjoint(split_households["validation"])


def test_load_training_data_reads_only_the_training_split(monkeypatch, tmp_path):
    train_dir = tmp_path / "train"
    validation_dir = tmp_path / "validation"
    train_dir.mkdir()
    validation_dir.mkdir()
    train_path = train_dir / "ch_load_features_1-1.parquet"
    validation_path = validation_dir / "ch_load_features_1-1.parquet"
    train_path.touch()
    validation_path.touch()
    loaded_paths = []
    monkeypatch.setattr(train_models, "TRAIN_FEATURE_DIR", train_dir)

    def fake_read_parquet(path):
        loaded_paths.append(path)
        return pd.DataFrame(
            {
                "profile_id": ["1-day"],
                "household_id": [1],
                "date": ["2023-01-01"],
                "feature": [0.5],
                "next_value": [1.0],
            }
        )

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    features, target, feature_columns = train_models.load_training_data()

    assert loaded_paths == [train_path]
    assert validation_path not in loaded_paths
    assert feature_columns == ["feature"]
    assert features["feature"].tolist() == [0.5]
    assert target.tolist() == [1.0]
