"""

v1: Genetically select best days to train models on, based on performance on the inner test set of the original data.

procedure:

- randomly create 100 sets without overlap/repetition
- score on inner test set
- keep best 50
- create 50 new sets with crossover + mutation
repeat

start with 10% mutation

create a csv file with worst, best and avg rsme performance for each cycle, call it rsme_evolution_{mutation_rate}pct_mutation.csv
create a png where we see avg rsme per cycle, name it the same way _.png
create a json with all the metadata, name it the same way _.json

create a csv with the 20 best sets and their rsme


we score by predicting loads of the global train set.


slop:
Genetically select strong day sets for training a base-load model.

This script optimizes a subset of valid household-day combinations by evolving
candidate sets and scoring them on a held-out validation sample. The search is
intentionally simple and deterministic given a seed, which makes it easy to
compare different day-set selections.

Procedure:
- build a search pool of valid household-day combinations from the SQLite data
- randomly create a population of unique day sets
- score each set on a fixed validation sample with a base-load XGBoost model
- keep the strongest elite sets
- generate new sets via crossover + mutation
- save the evolution metrics and the top candidates
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

# make project modules importable
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.simulation.controllers.mpc.predictors.ml.model_config import MODEL_FEATURES_BY_FAMILY  # noqa: E402
from training.features._regression import (  # noqa: E402
    _add_history_average_features,
    _add_std_features,
    _add_delta_features,
    _add_accel_feature,
    _round_float_features,
)
from training.features._shared import (  # noqa: E402
    _add_trig_time_features,
    _add_lag_features,
    _add_next_value_target,
)
from training.training_ch_load.day_sampling.sampling import RANDOM_TEST_FEATURE_DIR, SQLITE_PATH  # noqa: E402


FEATURE_COLUMNS = MODEL_FEATURES_BY_FAMILY["xgboost"]["base_load"]
TARGET_COLUMN = "next_value"
SEARCH_DIR = Path(__file__).parent / "genetic_selection_results"
FEATURE_TABLE_NAME = "load_features"


def _parse_candidate_limit(value: str | int | None) -> int | None:
    """Parse the candidate limit. None or 'none' disables the cap."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text in {"", "none", "null", "nan"}:
        return None
    parsed = int(text)
    if parsed < 0:
        raise argparse.ArgumentTypeError("candidate limit must be >= 0 or None")
    return parsed


def _choose_valid_day_pool(
    limit: int | None = None,
    start_ts: str = "2023-01-01 00:00:00",
    end_ts: str = "2024-12-30 23:45:00",
) -> list[tuple[int, str]]:
    """Return a list of valid (household_id, date) tuples."""
    with sqlite3.connect(SQLITE_PATH) as conn:
        raw_df = pd.read_sql("SELECT * FROM load", conn)

    raw_df = raw_df[raw_df["timestamp_utc"].between(start_ts, end_ts)].copy()
    raw_df["timestamp_utc"] = pd.to_datetime(raw_df["timestamp_utc"])
    raw_df["date"] = raw_df["timestamp_utc"].dt.date.astype(str)

    value_cols = [c for c in raw_df.columns if c not in {"timestamp_utc", "period", "date"}]
    melted = raw_df.melt(
        id_vars=["timestamp_utc", "period", "date"],
        value_vars=value_cols,
        var_name="household_id",
        value_name="load",
    )
    melted["household_id"] = melted["household_id"].astype(int)

    valid_days: list[tuple[int, str]] = []
    for (household_id, date), group in melted.groupby(["household_id", "date"], sort=False):
        if len(group) != 96 or group["load"].isna().any():
            continue
        valid_days.append((int(household_id), str(date)))

    if limit is not None and limit < len(valid_days):
        rng = random.Random(42)
        valid_days = rng.sample(valid_days, limit)

    # pool is built grouped by household; shuffle so pool index no longer
    # correlates with household ordering (avoids low-id bias in the GA).
    random.Random(1337).shuffle(valid_days)
    return valid_days


def _build_sample_frame_for_day_set(
    day_set: Sequence[tuple[int, str]],
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create one concatenated sample DataFrame that contains the chosen household-days."""
    if not day_set:
        raise ValueError("day_set must not be empty")

    keys = {(int(household_id), str(date)) for household_id, date in day_set}
    subset = raw_df[
        raw_df.apply(
            lambda row: (int(row["household_id"]), str(row["date"])) in keys,
            axis=1,
        )
    ].copy()

    if subset.empty:
        raise ValueError("No rows matched the requested day set")

    sample_df = subset[["timestamp_utc", "period", "household_id", "load"]].copy()
    sample_df["timestamp_utc"] = pd.to_datetime(sample_df["timestamp_utc"])
    sample_df["date"] = sample_df["timestamp_utc"].dt.date

    frames: list[pd.DataFrame] = []
    for _, day_df in sample_df.groupby("date", sort=False):
        if len(day_df) != 96 or day_df["load"].isna().any():
            continue

        player_day_df = pd.DataFrame(
            {
                "timestamp_utc": day_df["timestamp_utc"].to_numpy(),
                "timestep": day_df["period"].to_numpy(),
                "household_id": day_df["household_id"].to_numpy(),
                "base_load": day_df["load"].to_numpy(),
            }
        )

        player_day_df = _add_trig_time_features(player_day_df)
        player_day_df = _add_lag_features(
            player_day_df,
            source_column="base_load",
            group_cols=("household_id",),
            lags=(1, 2, 4, 8, 12),
            pad_value=-1.0,
            add_pad_flags=False,
            output_prefix="base_load_lag",
            dtype=float,
        )
        player_day_df = _add_history_average_features(
            player_day_df,
            windows=(2, 4, 8, 16),
            value_column="base_load",
            prefix="base_load",
        )
        player_day_df = _add_std_features(
            player_day_df,
            windows=(4, 8),
            value_column="base_load",
            prefix="base_load",
        )
        player_day_df = _add_delta_features(
            player_day_df,
            value_column="base_load",
            prefix="base_load",
        )
        player_day_df = _add_accel_feature(
            player_day_df,
            prefix="base_load",
        )
        player_day_df = _add_next_value_target(
            player_day_df,
            source_column="base_load",
            group_cols=("household_id",),
            target_column="next_value",
            fill_value=0.0,
            dtype=float,
        )
        player_day_df = _round_float_features(player_day_df, digits=3)
        frames.append(player_day_df)

    if not frames:
        raise ValueError("No valid feature frames could be built for the selected day set")

    return pd.concat(frames, ignore_index=True)


def _load_reference_test_df(test_seed: int = 0) -> pd.DataFrame:
    file_path = RANDOM_TEST_FEATURE_DIR / f"test_seed_{test_seed}_random_features.parquet"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Reference test sample not found: {file_path}. "
            "Generate it first with the sampling script."
        )
    return pd.read_parquet(file_path)


def _load_feature_rows_for_day_set(day_set: Sequence[tuple[int, str]]) -> pd.DataFrame:
    if not day_set:
        raise ValueError("day_set must not be empty")

    unique_pairs = list(dict.fromkeys((int(household_id), str(date)) for household_id, date in day_set))

    with sqlite3.connect(SQLITE_PATH) as conn:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (FEATURE_TABLE_NAME,),
        ).fetchone()
        if table_exists is None:
            raise FileNotFoundError(
                f"Feature table '{FEATURE_TABLE_NAME}' not found in {SQLITE_PATH}. "
                "Run data/dataset_CH/create_feature_table.py first."
            )

        pair_placeholders = ", ".join("(?, ?)" for _ in unique_pairs)
        query = (
            f"SELECT * FROM {FEATURE_TABLE_NAME} "
            f"WHERE (household_id, date) IN ({pair_placeholders})"
        )
        params = [value for pair in unique_pairs for value in pair]
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        raise ValueError(f"No feature rows found for day set: {day_set}")

    df = df.sort_values(["household_id", "date", "period"]).reset_index(drop=True)

    for col in ["household_id", "period", "timestep", "base_load", "time_sin", "time_cos"] + FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    return df


def _score_day_set(
    day_set: Sequence[tuple[int, str]],
    reference_df: pd.DataFrame,
    model_params: dict | None = None,
) -> float:
    train_df = _load_feature_rows_for_day_set(day_set)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required_columns if col not in train_df.columns]
    if missing:
        raise ValueError(f"Feature table is missing columns for scoring: {missing}")

    train_X = train_df[FEATURE_COLUMNS]
    train_y = train_df[TARGET_COLUMN]
    test_X = reference_df[FEATURE_COLUMNS]
    test_y = reference_df[TARGET_COLUMN]

    model = XGBRegressor(
        n_estimators=model_params.get("n_estimators", 300) if model_params else 300,
        max_depth=model_params.get("max_depth", 3) if model_params else 3,
        learning_rate=model_params.get("learning_rate", 0.05) if model_params else 0.05,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=1,
    )
    model.fit(train_X, train_y)
    preds = model.predict(test_X)
    return float(root_mean_squared_error(test_y, preds))


def _random_day_set(rng: random.Random, pool: Sequence[tuple[int, str]], n_days: int) -> list[int]:
    chosen = rng.sample(range(len(pool)), k=n_days)
    return sorted(chosen)


def _crossover(parent_a: Sequence[int], parent_b: Sequence[int], rng: random.Random) -> list[int]:
    if not parent_a or not parent_b:
        return []

    # sample unbiased from the gene union; a sorted-prefix cut would always
    # favor the smallest pool indices regardless of fitness.
    combined_genes = list(set(parent_a) | set(parent_b))
    rng.shuffle(combined_genes)
    target_size = min(len(combined_genes), max(len(parent_a), len(parent_b)))
    return sorted(combined_genes[:target_size])


def _mutate(chromosome: Sequence[int], pool_size: int, mutation_rate: float, rng: random.Random) -> list[int]:
    if not chromosome:
        return []

    mutated = list(chromosome)
    for i in range(len(mutated)):
        if rng.random() < mutation_rate:
            candidate = rng.randint(0, pool_size - 1)
            mutated[i] = candidate
    mutated = sorted(set(mutated))
    while len(mutated) < len(chromosome):
        mutated.append(rng.randint(0, pool_size - 1))
    return sorted(set(mutated))


def _evolve_day_sets(
    pool: Sequence[tuple[int, str]],
    raw_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    n_days: int,
    population_size: int,
    elite_count: int,
    generations: int,
    mutation_rate: float,
    seed: int,
) -> tuple[list[tuple[float, list[int]]], list[dict]]:
    rng = random.Random(seed)
    population: list[list[int]] = []
    population_keys: set[tuple[int, ...]] = set()
    while len(population) < population_size:
        chromosome = _random_day_set(rng, pool, n_days)
        chromosome_key = tuple(chromosome)
        if chromosome_key in population_keys:
            continue
        population.append(chromosome)
        population_keys.add(chromosome_key)

    history: list[dict] = []
    score_cache: dict[tuple[int, ...], float] = {}
    cache_hits = 0

    for generation in range(1, generations + 1):
        print(f"\n=== generation {generation}/{generations} ===")
        print(f"- initial population size: {len(population)}")
        print("- evaluating all candidates on validation set...")
        scores = []
        for idx, chromosome in enumerate(population, start=1):
            cache_key = tuple(chromosome)
            if cache_key in score_cache:
                rmse = score_cache[cache_key]
                cache_hits += 1
            else:
                day_set = [pool[idx_value] for idx_value in chromosome]
                rmse = _score_day_set(day_set, reference_df)
                score_cache[cache_key] = rmse
            scores.append((rmse, chromosome))
            if idx % max(1, len(population) // 5) == 0 or idx == len(population):
                print(f"  progress: {idx}/{len(population)} candidate sets scored")

        scores.sort(key=lambda item: item[0])
        best_rmse, worst_rmse = scores[0][0], scores[-1][0]
        avg_rmse = float(np.mean([value for value, _ in scores]))

        print(
            f"- generation summary: best={best_rmse:.6f}, avg={avg_rmse:.6f}, worst={worst_rmse:.6f}"
        )
        print(f"- score cache: {len(score_cache)} unique sets, {cache_hits} cache hits")

        history.append(
            {
                "generation": generation,
                "best_rmse": best_rmse,
                "avg_rmse": avg_rmse,
                "worst_rmse": worst_rmse,
            }
        )

        elites = [chromosome for _, chromosome in scores[:elite_count]]
        next_population = [list(chromosome) for chromosome in elites]
        next_population_keys = {tuple(chromosome) for chromosome in next_population}
        print(f"- selected elites: {len(elites)}")

        created = 0
        while len(next_population) < population_size:
            parent_a = rng.choice(elites)
            parent_b = rng.choice(elites)
            child = _crossover(parent_a, parent_b, rng)
            if len(child) < n_days:
                child.extend(rng.sample(range(len(pool)), k=n_days - len(child)))
            elif len(child) > n_days:
                child = sorted(rng.sample(child, n_days))

            child = _mutate(child, len(pool), mutation_rate, rng)
            if len(child) != n_days:
                child = sorted(set(child))
                while len(child) < n_days:
                    candidate = rng.randint(0, len(pool) - 1)
                    if candidate not in child:
                        child.append(candidate)
                if len(child) > n_days:
                    child = sorted(rng.sample(child, n_days))

            child_key = tuple(child)
            if child_key in next_population_keys:
                continue

            next_population.append(child)
            next_population_keys.add(child_key)
            created += 1
            if created % max(1, population_size // 5) == 0 or len(next_population) == population_size:
                print(f"  offspring built: {len(next_population)}/{population_size}")

        population = next_population
        print(f"- next generation prepared: {len(population)} sets")

    print("\n=== final evaluation ===")
    final_scores = []
    for idx, chromosome in enumerate(population, start=1):
        cache_key = tuple(chromosome)
        if cache_key in score_cache:
            rmse = score_cache[cache_key]
            cache_hits += 1
        else:
            day_set = [pool[idx_value] for idx_value in chromosome]
            rmse = _score_day_set(day_set, reference_df)
            score_cache[cache_key] = rmse
        final_scores.append((rmse, chromosome))
        if idx % max(1, len(population) // 5) == 0 or idx == len(population):
            print(f"  final scoring progress: {idx}/{len(population)}")

    final_scores.sort(key=lambda item: item[0])
    print(f"- final best RMSE: {final_scores[0][0]:.6f}")
    print(f"- total score cache: {len(score_cache)} unique sets, {cache_hits} cache hits")
    return final_scores, history


def _write_results(
    df_history: pd.DataFrame,
    best_sets: pd.DataFrame,
    output_dir: Path,
    mutation_rate: float,
    n_days: int,
    population_size: int,
    generations: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"rmse_evolution_n_days_{n_days}_pop_{population_size}_gen_{generations}_"
        f"mut_{mutation_rate}_seed_{seed}"
    )

    history_path = output_dir / f"{stem}.csv"
    plot_path = output_dir / f"{stem}.png"
    metadata_path = output_dir / f"{stem}.json"
    best_path = output_dir / f"{stem}_top_sets.csv"

    df_history.to_csv(history_path, index=False)
    best_sets.to_csv(best_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_history["generation"], df_history["best_rmse"], label="best", marker="o")
    ax.plot(df_history["generation"], df_history["avg_rmse"], label="avg", linestyle="--")
    ax.plot(df_history["generation"], df_history["worst_rmse"], label="worst", linestyle=":")
    ax.set_xlabel("generation")
    ax.set_ylabel("RMSE")
    ax.set_title("Genetic day-set selection evolution")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    metadata = {
        "n_days": n_days,
        "population_size": population_size,
        "elite_count": max(1, population_size // 2),
        "generations": generations,
        "mutation_rate": mutation_rate,
        "seed": seed,
        "output_files": {
            "history_csv": history_path.name,
            "plot_png": plot_path.name,
            "top_sets_csv": best_path.name,
        },
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _prepare_best_sets(
    scored_sets: list[tuple[float, list[int]]],
    pool: Sequence[tuple[int, str]],
    n_top: int = 20,
) -> pd.DataFrame:
    rows = []
    seen_sets: set[tuple[int, ...]] = set()
    for score, chromosome in scored_sets[:n_top]:
        set_key = tuple(chromosome)
        if set_key in seen_sets:
            continue
        seen_sets.add(set_key)
        day_pairs = [pool[idx] for idx in chromosome]
        rows.append(
            {
                "rmse": float(score),
                "n_days": len(day_pairs),
                "day_set": ";".join(f"{household_id}_{date}" for household_id, date in day_pairs),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genetically select the best day sets for model training.")
    parser.add_argument("--n-days", type=int, default=150, help="How many household-day samples each set contains.")
    parser.add_argument("--population-size", type=int, default=100, help="Number of candidate day sets per generation.")
    parser.add_argument("--elite-count", type=int, default=50, help="Number of elite sets kept from each generation.")
    parser.add_argument("--generations", type=int, default=25, help="Number of evolutionary iterations.")
    parser.add_argument("--mutation-rate", type=float, default=0.10, help="Mutation probability used per gene.")
    parser.add_argument(
        "--candidate-limit",
        type=_parse_candidate_limit,
        default=None,
        help="Maximum number of valid day combinations to consider; use None/none to disable the cap.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible optimization.")
    parser.add_argument("--test-seed", type=int, default=0, help="Reference validation sample seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SEARCH_DIR,
        help="Folder for the CSV, PNG and JSON files generated by the optimization.",
    )
    args = parser.parse_args()

    if args.n_days <= 0:
        raise ValueError("--n-days must be positive")
    if args.population_size <= 0:
        raise ValueError("--population-size must be positive")
    if args.elite_count <= 0:
        raise ValueError("--elite-count must be positive")
    if args.generations <= 0:
        raise ValueError("--generations must be positive")
    if not 0.0 <= args.mutation_rate <= 1.0:
        raise ValueError("--mutation-rate must be between 0 and 1")

    print(f"Building valid day pool with candidate_limit={args.candidate_limit}")
    pool = _choose_valid_day_pool(limit=args.candidate_limit)
    print(f"Valid day pool size: {len(pool)}")
    if len(pool) < args.n_days:
        raise ValueError(
            f"This dataset only contains {len(pool)} valid day combinations, "
            f"but --n-days requires {args.n_days}."
        )

    reference_df = _load_reference_test_df(test_seed=args.test_seed)
    scored_sets, history = _evolve_day_sets(
        pool=pool,
        raw_df=pd.DataFrame(),
        reference_df=reference_df,
        n_days=args.n_days,
        population_size=args.population_size,
        elite_count=min(args.elite_count, args.population_size),
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
    )

    df_history = pd.DataFrame(history)
    best_sets = _prepare_best_sets(scored_sets, pool, n_top=20)
    _write_results(
        df_history=df_history,
        best_sets=best_sets,
        output_dir=args.output_dir,
        mutation_rate=args.mutation_rate,
        n_days=args.n_days,
        population_size=args.population_size,
        generations=args.generations,
        seed=args.seed,
    )

    best_rmse, best_chromosome = scored_sets[0]
    best_day_set = [pool[idx] for idx in best_chromosome]
    print("\n=== optimization finished ===")
    print(f"Best RMSE found: {best_rmse:.6f}")
    print(f"Best day set contains {len(best_day_set)} household-day combinations")
    print(f"Top entries saved to: {args.output_dir}")
    print("Sample of best set:")
    for household_id, date in best_day_set[:10]:
        print(f"  household_id={household_id}, date={date}")


if __name__ == "__main__":
    main()
