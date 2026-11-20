from collections.abc import Callable, Mapping


StepPredictor = Callable[[int, float, list[float]], float]
Bypass = Callable[[int, list[float], float], float | None]


def _apply_bypasses(
    values: list[float],
    current_timestep: int,
    history: list[float],
    bypass: Bypass | None,
) -> list[float]:
    if bypass is None:
        return values

    result = list(values)
    for offset in range(1, len(result)):
        bypass_value = bypass(
            current_timestep + offset - 1,
            history + result[: offset - 1],
            result[offset - 1],
        )
        if bypass_value is not None:
            result[offset] = bypass_value
    return result


def smooth_recursive_interpolate(
    anchors: Mapping[int, float],
    horizon: int,
    current_timestep: int,
    history: list[float],
    step_predictor: StepPredictor,
    bypass: Bypass | None = None,
) -> list[float]:
    if horizon <= 0:
        return []
    if 0 not in anchors:
        raise ValueError("Anchored recursive interpolation requires an anchor at 0.")

    bounded_anchors = sorted(
        (offset, float(value))
        for offset, value in anchors.items()
        if 0 <= offset < horizon
    )
    values = [float(anchors[0])]

    def predict_next(current_offset: int, current_value: float, prior: list[float]) -> float:
        timestep = current_timestep + current_offset
        if bypass is not None:
            bypass_value = bypass(timestep, history + prior, current_value)
            if bypass_value is not None:
                return float(bypass_value)
        return float(
            step_predictor(timestep, current_value, history + prior)
        )

    for (left, _), (right, right_anchor) in zip(
        bounded_anchors,
        bounded_anchors[1:],
    ):
        span = right - left
        if span == 1:
            values.append(right_anchor)
            continue

        raw_segment = [values[left]]
        for relative_offset in range(1, span + 1):
            current_offset = left + relative_offset - 1
            prior = values[:left] + raw_segment[: relative_offset - 1]
            raw_segment.append(
                predict_next(
                    current_offset,
                    raw_segment[-1],
                    prior,
                )
            )

        correction = right_anchor - raw_segment[-1]
        for relative_offset in range(1, span + 1):
            alpha = relative_offset / span
            values.append(raw_segment[relative_offset] + alpha * correction)

    last_anchor = bounded_anchors[-1][0]
    for next_offset in range(last_anchor + 1, horizon):
        current_offset = next_offset - 1
        values.append(
            predict_next(
                current_offset,
                values[-1],
                values[:current_offset],
            )
        )

    return _apply_bypasses(
        values,
        current_timestep,
        history,
        bypass,
    )
