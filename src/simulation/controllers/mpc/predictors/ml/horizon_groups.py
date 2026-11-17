HORIZON_GROUPS: tuple[tuple[str, int, int], ...] = (
    ("h1_12", 1, 12),
    ("h13_42", 13, 42),
    ("h43_95", 43, 95),
)


def horizon_group_name(prediction_horizon: int) -> str:
    for name, first_horizon, last_horizon in HORIZON_GROUPS:
        if first_horizon <= prediction_horizon <= last_horizon:
            return name
    raise ValueError(
        f"No horizon group is configured for prediction horizon {prediction_horizon}."
    )
