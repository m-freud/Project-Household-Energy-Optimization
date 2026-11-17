import pandas as pd

from training.features.feature_parquets._split_utils import (
    assign_profiles_to_splits,
    split_household_ids,
)
from training.features.feature_parquets.ev2 import create_train_features as ev2_features
from training.features.feature_parquets.pv import create_train_features as pv_features


def test_household_id_split_is_deterministic_disjoint_and_approximately_80_10_10():
    household_ids = [f"household-{index}" for index in range(100)]

    first_split = split_household_ids(household_ids, random_seed=42)
    second_split = split_household_ids(household_ids, random_seed=42)

    assert first_split == second_split
    assert {split: len(ids) for split, ids in first_split.items()} == {
        "train": 80,
        "test": 10,
        "validation": 10,
    }
    assert set(first_split["train"]).isdisjoint(first_split["test"])
    assert set(first_split["train"]).isdisjoint(first_split["validation"])
    assert set(first_split["test"]).isdisjoint(first_split["validation"])
    assert set.union(*(set(ids) for ids in first_split.values())) == set(household_ids)


def test_profiles_from_the_same_household_are_assigned_to_the_same_split():
    household_splits = {
        "train": ["train-household"],
        "test": ["test-household"],
        "validation": ["validation-household"],
    }

    profiles_by_split = assign_profiles_to_splits(
        ["train-profile-a", "test-profile", "train-profile-b", "validation-profile"],
        {
            "train-profile-a": "train-household",
            "test-profile": "test-household",
            "train-profile-b": "train-household",
            "validation-profile": "validation-household",
        },
        household_splits,
    )

    assert profiles_by_split == {
        "train": ["train-profile-a", "train-profile-b"],
        "test": ["test-profile"],
        "validation": ["validation-profile"],
    }


def test_ev2_features_preserve_source_household_id():
    profile = pd.DataFrame(
        {
            "period": range(1, ev2_features.PERIODS_PER_DAY + 1),
            "7": [0] * ev2_features.PERIODS_PER_DAY,
        }
    )

    features = ev2_features._make_profile_features(
        profile,
        "7",
        {"profile_id": "person-profile", "household_id": "source-household"},
    )

    assert set(features["profile_id"]) == {"person-profile"}
    assert set(features["household_id"]) == {"source-household"}


def test_pv_split_reserves_runtime_and_uses_disjoint_days_for_test_and_validation():
    players = ["residential1", "residential3", "residential4", "residential6"]
    profiles = [
        (player, f"day-{day}")
        for player in players
        for day in range(20)
    ]

    first_split = pv_features._split_profiles_by_player(profiles, players)
    second_split = pv_features._split_profiles_by_player(profiles, players)

    assert first_split == second_split
    assert len({player for player, _ in first_split["train"]}) == 2
    assert len({player for player, _ in first_split["test"]}) == 1
    assert len({player for player, _ in first_split["validation"]}) == 1
    assert "residential3" not in {
        player for split_profiles in first_split.values() for player, _ in split_profiles
    }
    test_player = {player for player, _ in first_split["test"]}
    validation_player = {player for player, _ in first_split["validation"]}
    assert test_player == validation_player
    assert {date for _, date in first_split["test"]}.isdisjoint(
        date for _, date in first_split["validation"]
    )
    train_players = {player for player, _ in first_split["train"]}
    assert train_players.isdisjoint(test_player)
