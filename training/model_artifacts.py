from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence


def render_training_params_manifest(
    *,
    family: str,
    target: str,
    fold_ids: Sequence[str],
    params: Mapping[str, object],
) -> str:
    lines = [
        f"family: {family}",
        f"target: {target}",
        f"fold_ids: {', '.join(fold_ids)}",
        "params:",
    ]
    for key in sorted(params.keys()):
        lines.append(f"  {key}: {params[key]}")
    lines.append("")
    return "\n".join(lines)


def write_training_params_manifest(
    model_dir: Path,
    *,
    family: str,
    target: str,
    fold_ids: Sequence[str],
    params: Mapping[str, object],
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "training_params.txt"
    manifest_path.write_text(
        render_training_params_manifest(
            family=family,
            target=target,
            fold_ids=fold_ids,
            params=params,
        ),
        encoding="utf-8",
    )
    return manifest_path