"""Generate training/split/test_folds.csv from Config constants.

Rules:
- fold_members come from Config.CLEAN_SUBSETS
- fold_complement comes from Config.CLEAN_COMPLEMENTS
- global complement for each metric is fold_complement + Config.EXTRA_IDS[metric]
"""

from __future__ import annotations

from pathlib import Path
import csv
import sys


# Enable imports from src/
cwd = Path.cwd().resolve()
repo_root = next((p for p in [cwd, *cwd.parents] if (p / "src").exists()), cwd)
sys.path.insert(0, str(repo_root))

from src.config import Config


def _format_ids(ids: list[int]) -> str:
    return ", ".join(str(player_id) for player_id in ids)


def _build_global_complement(fold_complement: list[int], metric: str) -> list[int]:
    # Use set + sort to match Config.TRAIN_SETS behavior.
    return sorted(set(fold_complement + Config.EXTRA_IDS[metric]))


def main() -> None:
    output_path = repo_root / "training" / "split" / "test_folds.csv"

    metric_columns = [
        "base_load",
        "pv_gen",
        "ev1_status",
        "ev2_status",
    ]

    fieldnames = [
        "fold_id",
        "fold_members",
        "fold_complement",
        *[f"global_complement_{metric}" for metric in metric_columns],
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for fold_id in sorted(Config.CLEAN_SUBSETS.keys()):
            fold_members = list(Config.CLEAN_SUBSETS[fold_id])
            fold_complement = list(Config.CLEAN_COMPLEMENTS[fold_id])

            row = {
                "fold_id": fold_id,
                "fold_members": _format_ids(fold_members),
                "fold_complement": _format_ids(fold_complement),
            }

            for metric in metric_columns:
                global_complement = _build_global_complement(fold_complement, metric)
                row[f"global_complement_{metric}"] = _format_ids(global_complement)

            writer.writerow(row)

    print(f"written {output_path}")


if __name__ == "__main__":
    main()
