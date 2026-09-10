"""
v1: Genetically select best days to train a base-load model on, based on
simulated final net cost (instead of RMSE on a held-out validation sample).

Same evolutionary procedure as genetic_selection.py:
- randomly create a population of unique day sets
- score each set by training an XGBoost model on it, then running a full
  MPC simulation (everything but base_load predicted by the oracle) over a
  fixed household test set and taking the mean net cost
- keep the strongest elites (lowest net cost)
- create new sets via crossover + mutation
- repeat

XGBoost hyperparameters are fixed (no sweep here):
{"learning_rate": 0.02, "max_depth": 4, "n_estimators": 100}

--test-set selects which household partition is simulated on:
- "test"     -> PARTITIONS["global"]["test"]
- "train"    -> PARTITIONS["global"]["train"]
- "20_loads" -> RuntimeConfig.INDEPENDENT_TEST_SET_20

Outputs (same shape as genetic_selection.py, net_cost instead of rmse):
- net_cost_evolution_{...}.csv: best/avg/worst net cost per generation
- net_cost_evolution_{...}.png: plot of the same
- net_cost_evolution_{...}.json: run metadata
- net_cost_evolution_{...}_top_sets.csv: the 20 best day sets and their net cost
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# make project modules importable
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from simulation.controllers.mpc.predictors.ml.recursive.recursive_ml_predictor import RecursiveMLPredictor  # noqa: E402
from src.simulation.controllers.mpc.predictors.ml.model_config import MODEL_FEATURES_BY_FAMILY  # noqa: E402
from src.simulation.controllers.mpc.predictors.modular_predictor import ModularPredictor  # noqa: E402
from src.simulation.controllers.mpc.predictors.oracle.oracle_predictor import OraclePredictor  # noqa: E402
from src.simulation.run_context import RunContext  # noqa: E402
from src.simulation.scenarios.scenario import scenarios as scenario_catalog  # noqa: E402
from src.simulation.simulation import Simulation  # noqa: E402
from src.runtime_config import RuntimeConfig  # noqa: E402
from src.sqlite_connection import sqlite_conn  # noqa: E402
from training.split.clean_split import PARTITIONS  # noqa: E402
from training.training_ch_load.day_sampling.sampling import SQLITE_PATH  # noqa: E402


FEATURE_COLUMNS = MODEL_FEATURES_BY_FAMILY["xgboost"]["base_load"]
TARGET_COLUMN = "next_value"
TARGET = "base_load"
MODEL_PARAMS = {"learning_rate": 0.02, "max_depth": 4, "n_estimators": 100}
SEARCH_DIR = Path(__file__).parent / "genetic_selection_net_cost_results"
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


def _resolve_test_household_ids(test_set: str, n_test_ids: int | None) -> list[int]:
    """Map --test-set to a fixed household partition (Portugal households)."""
    if test_set == "20_loads":
        household_ids = list(RuntimeConfig.INDEPENDENT_TEST_SET_20)
    elif test_set in {"test", "train"}:
        household_ids = list(PARTITIONS["global"][test_set])
    else:
        raise ValueError(f"--test-set must be 'test', 'train' or '20_loads', got: {test_set}")
    if n_test_ids is not None:
        household_ids = household_ids[:n_test_ids]
    return household_ids


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
    test_household_ids: Sequence[int],
    scenario_name: str,
) -> float:
    """Train an XGBoost model on the day set and return the simulated mean net cost."""
    train_df = _load_feature_rows_for_day_set(day_set)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required_columns if col not in train_df.columns]
    if missing:
        raise ValueError(f"Feature table is missing columns for scoring: {missing}")

    train_X = train_df[FEATURE_COLUMNS]
    train_y = train_df[TARGET_COLUMN]

    model = XGBRegressor(
        n_estimators=MODEL_PARAMS["n_estimators"],
        max_depth=MODEL_PARAMS["max_depth"],
        learning_rate=MODEL_PARAMS["learning_rate"],
        random_state=42,
        objective="reg:squarederror",
        n_jobs=1,
    )
    model.fit(train_X, train_y)

    predictor = ModularPredictor(
        default_predictor=OraclePredictor(),
        target_predictors={
            TARGET: RecursiveMLPredictor(
                base_load_model=model,
                pv_gen_model=None,
                ev1_status_model=None,
                ev2_status_model=None,
            )
        },
    )
    simulation = Simulation(sqlite_conn, ensure_results_table=False)
    run_context = RunContext(
        controller_factory=simulation.make_mpc_controller("mpc_iso_benchmark", 96, predictor=predictor),
        controller_name="mpc_iso_benchmark",
        scenario=scenario_catalog[scenario_name],
        start_time=1,
    )
    results = simulation.run_batch(
        run_contexts=[run_context],
        household_ids=list(test_household_ids),
        parallel_households=True,
        parallel_workers=6,
        write_results_to_sqlite=False,
    )
    return float(np.mean(results["net_costs"]))


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
    test_household_ids: Sequence[int],
    scenario_name: str,
    n_days: int,
    population_size: int,
    elite_count: int,
    generations: int,
    mutation_rate: float,
    seed: int,
    history_path: Path,
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
    started_at = time.perf_counter()

    for generation in range(1, generations + 1):
        print(f"\n=== generation {generation}/{generations} ===")
        print(f"- initial population size: {len(population)}")
        print("- evaluating all candidates via simulated net cost...")
        scores = []
        for idx, chromosome in enumerate(population, start=1):
            cache_key = tuple(chromosome)
            if cache_key in score_cache:
                net_cost = score_cache[cache_key]
                cache_hits += 1
            else:
                day_set = [pool[idx_value] for idx_value in chromosome]
                net_cost = _score_day_set(day_set, test_household_ids, scenario_name)
                score_cache[cache_key] = net_cost
            scores.append((net_cost, chromosome))
            if idx % max(1, len(population) // 5) == 0 or idx == len(population):
                elapsed = time.perf_counter() - started_at
                print(f"  progress: {idx}/{len(population)} candidate sets scored (elapsed={elapsed/60:.1f}m)")

        scores.sort(key=lambda item: item[0])
        best_net_cost, worst_net_cost = scores[0][0], scores[-1][0]
        avg_net_cost = float(np.mean([value for value, _ in scores]))

        print(
            f"- generation summary: best={best_net_cost:.6f}, avg={avg_net_cost:.6f}, worst={worst_net_cost:.6f}"
        )
        print(f"- score cache: {len(score_cache)} unique sets, {cache_hits} cache hits")

        generation_row = {
            "generation": generation,
            "best_net_cost": best_net_cost,
            "avg_net_cost": avg_net_cost,
            "worst_net_cost": worst_net_cost,
        }
        history.append(generation_row)
        pd.DataFrame([generation_row]).to_csv(history_path, mode="a", header=False, index=False)

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

        elapsed = time.perf_counter() - started_at
        avg_per_generation = elapsed / generation
        eta = avg_per_generation * (generations - generation)
        print(
            f"- progress: {generation}/{generations} generations, "
            f"elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m"
        )

    print("\n=== final evaluation ===")
    final_scores = []
    for idx, chromosome in enumerate(population, start=1):
        cache_key = tuple(chromosome)
        if cache_key in score_cache:
            net_cost = score_cache[cache_key]
            cache_hits += 1
        else:
            day_set = [pool[idx_value] for idx_value in chromosome]
            net_cost = _score_day_set(day_set, test_household_ids, scenario_name)
            score_cache[cache_key] = net_cost
        final_scores.append((net_cost, chromosome))
        if idx % max(1, len(population) // 5) == 0 or idx == len(population):
            print(f"  final scoring progress: {idx}/{len(population)}")

    final_scores.sort(key=lambda item: item[0])
    print(f"- final best net cost: {final_scores[0][0]:.6f}")
    print(f"- total score cache: {len(score_cache)} unique sets, {cache_hits} cache hits")
    return final_scores, history


def _write_results(
    df_history: pd.DataFrame,
    best_sets: pd.DataFrame,
    output_dir: Path,
    mutation_rate: float,
    n_days: int,
    population_size: int,
    elite_count: int,
    generations: int,
    seed: int,
    test_set: str,
    scenario_name: str,
) -> None:
    stem = (
        f"net_cost_evolution_n_days_{n_days}_pop_{population_size}_gen_{generations}_"
        f"mut_{mutation_rate}_seed_{seed}_testset_{test_set}"
    )

    # history csv is written incrementally (row per generation) during _evolve_day_sets
    history_path = output_dir / f"{stem}.csv"
    plot_path = output_dir / f"{stem}.png"
    metadata_path = output_dir / f"{stem}.json"
    best_path = output_dir / f"{stem}_top_sets.csv"

    best_sets.to_csv(best_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_history["generation"], df_history["best_net_cost"], label="best", marker="o")
    ax.plot(df_history["generation"], df_history["avg_net_cost"], label="avg", linestyle="--")
    ax.plot(df_history["generation"], df_history["worst_net_cost"], label="worst", linestyle=":")
    ax.set_xlabel("generation")
    ax.set_ylabel("net cost")
    ax.set_title("Genetic day-set selection evolution (simulated net cost)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    metadata = {
        "n_days": n_days,
        "population_size": population_size,
        "elite_count": elite_count,
        "generations": generations,
        "mutation_rate": mutation_rate,
        "seed": seed,
        "test_set": test_set,
        "scenario": scenario_name,
        "model_params": MODEL_PARAMS,
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
                "net_cost": float(score),
                "n_days": len(day_pairs),
                "day_set": ";".join(f"{household_id}_{date}" for household_id, date in day_pairs),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genetically select the best day sets for base-load model training, "
        "scored by simulated net cost."
    )
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
    parser.add_argument(
        "--test-set",
        type=str,
        choices=["test", "train", "20_loads"],
        default="test",
        help="Which household partition to simulate net cost on: "
        "'test' -> PARTITIONS['global']['test'], 'train' -> PARTITIONS['global']['train'], "
        "'20_loads' -> RuntimeConfig.INDEPENDENT_TEST_SET_20.",
    )
    parser.add_argument(
        "--n-test-ids",
        type=int,
        default=None,
        help="Optionally cap the number of simulated households, for faster (noisier) scoring.",
    )
    parser.add_argument("--scenario", type=str, default="default_scenario", help="Simulation scenario to score on.")
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

    test_household_ids = _resolve_test_household_ids(args.test_set, args.n_test_ids)
    print(f"Simulating net cost on {len(test_household_ids)} households (test_set={args.test_set})")

    resolved_elite_count = min(args.elite_count, args.population_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"net_cost_evolution_n_days_{args.n_days}_pop_{args.population_size}_gen_{args.generations}_"
        f"mut_{args.mutation_rate}_seed_{args.seed}_testset_{args.test_set}"
    )
    history_path = args.output_dir / f"{stem}.csv"
    pd.DataFrame(columns=["generation", "best_net_cost", "avg_net_cost", "worst_net_cost"]).to_csv(
        history_path, index=False
    )  # init file early so partial progress survives crashes

    scored_sets, history = _evolve_day_sets(
        pool=pool,
        test_household_ids=test_household_ids,
        scenario_name=args.scenario,
        n_days=args.n_days,
        population_size=args.population_size,
        elite_count=resolved_elite_count,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
        history_path=history_path,
    )

    df_history = pd.DataFrame(history)
    best_sets = _prepare_best_sets(scored_sets, pool, n_top=min(20, args.population_size))
    _write_results(
        df_history=df_history,
        best_sets=best_sets,
        output_dir=args.output_dir,
        mutation_rate=args.mutation_rate,
        n_days=args.n_days,
        population_size=args.population_size,
        elite_count=resolved_elite_count,
        generations=args.generations,
        seed=args.seed,
        test_set=args.test_set,
        scenario_name=args.scenario,
    )

    best_net_cost, best_chromosome = scored_sets[0]
    best_day_set = [pool[idx] for idx in best_chromosome]
    print("\n=== optimization finished ===")
    print(f"Best net cost found: {best_net_cost:.6f}")
    print(f"Best day set contains {len(best_day_set)} household-day combinations")
    print(f"Top entries saved to: {args.output_dir}")
    print("Sample of best set:")
    for household_id, date in best_day_set[:10]:
        print(f"  household_id={household_id}, date={date}")


if __name__ == "__main__":
    main()
