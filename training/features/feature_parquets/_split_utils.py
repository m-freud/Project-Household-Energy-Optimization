import random


def split_household_ids(
    household_ids: list[str] | set[str],
    *,
    random_seed: int,
) -> dict[str, list[str]]:
    unique_household_ids = sorted(set(household_ids))
    if len(unique_household_ids) < 3:
        raise ValueError(
            "At least three distinct household IDs are required for "
            "train, test, and validation splits"
        )

    shuffled_household_ids = random.Random(random_seed).sample(
        unique_household_ids,
        len(unique_household_ids),
    )
    test_count = max(1, round(len(unique_household_ids) * 0.1))
    validation_count = max(1, round(len(unique_household_ids) * 0.1))
    if test_count + validation_count >= len(unique_household_ids):
        raise ValueError(
            "Not enough distinct household IDs remain for a non-empty training split"
        )

    test_ids = shuffled_household_ids[:test_count]
    validation_ids = shuffled_household_ids[test_count : test_count + validation_count]
    training_ids = shuffled_household_ids[test_count + validation_count :]
    return {
        "train": training_ids,
        "test": test_ids,
        "validation": validation_ids,
    }


def assign_profiles_to_splits(
    profile_ids: list[str],
    household_id_by_profile: dict[str, str],
    household_splits: dict[str, list[str]],
) -> dict[str, list[str]]:
    profiles_by_split: dict[str, list[str]] = {}
    for split, household_ids in household_splits.items():
        allowed_households = set(household_ids)
        profiles_by_split[split] = [
            profile_id
            for profile_id in profile_ids
            if household_id_by_profile[profile_id] in allowed_households
        ]
    return profiles_by_split
