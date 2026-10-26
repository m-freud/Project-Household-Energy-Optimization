"""
Create a runtime stratified sample from CH base_load_pool.

Workflow:
1. Load the original energy.db distribution for max_load vs total_energy.
2. Create 2D quantile bins from the original metadata.
3. Compute the relative frequency of each 2D bin in the original data.
4. Sample 250 profiles from CH_loads.db.base_load_pool with the same bin shares.
5. Prefer household/player IDs that are already in use.
6. Save the runtime sample to CH_loads.db.
7. Report how many unique CH player IDs remain outside the runtime sample.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CH_DB_PATH = ROOT / "sqlite" / "staging" / "CH_loads.db"
ENERGY_DB_PATH = ROOT / "sqlite" / "energy.db"
OUTPUT_DIR = Path(__file__).resolve().parent
TARGET_RUNTIME_SIZE = 250
N_QUANTILE_BINS = 10


def load_original_metadata() -> pd.DataFrame:
    with sqlite3.connect(ENERGY_DB_PATH) as conn:
        return pd.read_sql(
            "SELECT player_id, total_energy, max_load FROM base_load_metadata",
            conn,
        )


def load_ch_pool_metadata() -> pd.DataFrame:
    with sqlite3.connect(CH_DB_PATH) as conn:
        return pd.read_sql(
            "SELECT player_id, date, total_energy, max_load FROM base_load_pool",
            conn,
        )


def load_preferred_player_ids() -> set[str]:
    preferred = set()
    with sqlite3.connect(CH_DB_PATH) as conn:
        for table in ["base_load_runtime_profiles", "base_load_training_profiles", "base_load_stratified_sample"]:
            try:
                rows = pd.read_sql(
                    f"SELECT DISTINCT player_id FROM {table}",
                    conn,
                )
            except Exception:
                continue
            preferred.update(rows["player_id"].astype(str).tolist())
    return preferred


def compute_bin_edges(df: pd.DataFrame) -> dict[str, np.ndarray]:
    edges: dict[str, np.ndarray] = {}
    for col in ["max_load", "total_energy"]:
        edges[col] = np.quantile(df[col], np.linspace(0, 1, N_QUANTILE_BINS + 1))
    return edges


def add_bin_columns(df: pd.DataFrame, bin_edges: dict[str, np.ndarray]) -> pd.DataFrame:
    out = df.copy()
    out["max_load_bin"] = pd.cut(
        out["max_load"],
        bins=bin_edges["max_load"],
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )
    out["total_energy_bin"] = pd.cut(
        out["total_energy"],
        bins=bin_edges["total_energy"],
        labels=False,
        include_lowest=True,
        duplicates="drop",
    )
    return out


def build_distribution(df: pd.DataFrame) -> pd.DataFrame:
    dist = (
        df.groupby(["max_load_bin", "total_energy_bin"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    dist["relative_frequency"] = dist["count"] / dist["count"].sum()
    return dist


def largest_remainder_allocation(dist: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    dist = dist.copy()
    dist["target_raw"] = dist["relative_frequency"] * sample_size
    dist["target"] = np.floor(dist["target_raw"]).astype(int)
    dist["fractional_part"] = dist["target_raw"] - dist["target"]

    remaining = sample_size - dist["target"].sum()
    if remaining > 0:
        for idx in dist.sort_values("fractional_part", ascending=False).head(remaining).index:
            dist.at[idx, "target"] += 1

    return dist


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    corr = df[["max_load", "total_energy"]].corr().iloc[0, 1]
    rows = [
        [df["max_load"].mean(), df["total_energy"].mean()],
        [df["max_load"].std(ddof=1), df["total_energy"].std(ddof=1)],
        [df["max_load"].median(), df["total_energy"].median()],
        [df["max_load"].quantile(0.95), df["total_energy"].quantile(0.95)],
        [corr, corr],
    ]
    return pd.DataFrame(
        rows,
        index=["mean", "std", "median", "p95", "corr(max_load,total_energy)"],
        columns=["max_load", "total_energy"],
    )


def sample_runtime_profiles(ch_pool: pd.DataFrame, original_bins: pd.DataFrame, preferred_ids: set[str]) -> pd.DataFrame:
    target_dist = largest_remainder_allocation(original_bins, TARGET_RUNTIME_SIZE)
    selected_frames: list[pd.DataFrame] = []
    remaining_pool = ch_pool.copy()

    for _, row in target_dist.sort_values("target", ascending=False).iterrows():
        requested = int(row["target"])
        if requested <= 0:
            continue

        bin_key = (int(row["max_load_bin"]), int(row["total_energy_bin"]))
        bin_df = remaining_pool[
            (remaining_pool["max_load_bin"] == bin_key[0])
            & (remaining_pool["total_energy_bin"] == bin_key[1])
        ].copy()

        if bin_df.empty:
            print(f"Warning: no CH profiles left in bin {bin_key}; continuing with remaining pool.")
            continue

        preferred_df = bin_df[bin_df["player_id"].astype(str).isin(preferred_ids)].copy()
        fallback_df = bin_df[~bin_df["player_id"].astype(str).isin(preferred_ids)].copy()
        ordered = pd.concat([preferred_df, fallback_df], axis=0, ignore_index=False)

        take_n = min(requested, len(ordered))
        chosen = ordered.iloc[:take_n].copy()
        if len(chosen) == 0:
            continue

        chosen["sample_bin"] = f"{bin_key[0]}_{bin_key[1]}"
        selected_frames.append(chosen)
        remaining_pool = remaining_pool.drop(index=chosen.index)

    if not selected_frames:
        raise ValueError("No runtime profiles could be selected from the CH pool.")

    sample_df = pd.concat(selected_frames, ignore_index=True)
    sample_df = sample_df.head(TARGET_RUNTIME_SIZE).reset_index(drop=True)
    sample_df = sample_df[["player_id", "date", "total_energy", "max_load", "sample_bin"]].copy()
    return sample_df


def save_runtime_sample(sample_df: pd.DataFrame) -> None:
    with sqlite3.connect(CH_DB_PATH) as conn:
        sample_df.to_sql("base_load_runtime_profiles", conn, if_exists="replace", index=False)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_base_load_runtime_profiles_player_date "
            "ON base_load_runtime_profiles (player_id, date)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_base_load_runtime_profiles_max_load "
            "ON base_load_runtime_profiles (max_load)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_base_load_runtime_profiles_total_energy "
            "ON base_load_runtime_profiles (total_energy)"
        )


def save_training_sample(training_df: pd.DataFrame) -> None:
    with sqlite3.connect(CH_DB_PATH) as conn:
        training_df.to_sql("base_load_training_profiles", conn, if_exists="replace", index=False)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_base_load_training_profiles_player_date "
            "ON base_load_training_profiles (player_id, date)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_base_load_training_profiles_max_load "
            "ON base_load_training_profiles (max_load)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_base_load_training_profiles_total_energy "
            "ON base_load_training_profiles (total_energy)"
        )


def plot_runtime_vs_original(original_df: pd.DataFrame, sample_df: pd.DataFrame) -> None:
    x_all = pd.concat([original_df["total_energy"], sample_df["total_energy"]])
    y_all = pd.concat([original_df["max_load"], sample_df["max_load"]])
    x_edges = np.quantile(x_all, np.linspace(0, 1, N_QUANTILE_BINS + 1))
    y_edges = np.quantile(y_all, np.linspace(0, 1, N_QUANTILE_BINS + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax in axes:
        ax.set_xlabel("total_energy")
        ax.set_ylabel("max_load")

    hist_original = axes[0].hist2d(original_df["total_energy"], original_df["max_load"], bins=[x_edges, y_edges], cmap="Blues")
    axes[0].set_title("Original energy.db distribution")
    fig.colorbar(hist_original[3], ax=axes[0], label="count")

    hist_sample = axes[1].hist2d(sample_df["total_energy"], sample_df["max_load"], bins=[x_edges, y_edges], cmap="Oranges")
    axes[1].set_title(f"Runtime CH sample (n={len(sample_df)})")
    fig.colorbar(hist_sample[3], ax=axes[1], label="count")

    out_path = OUTPUT_DIR / "runtime_sample_2d_hist.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved 2D histogram: {out_path}")


def plot_training_vs_original(original_df: pd.DataFrame, training_df: pd.DataFrame) -> None:
    x_all = pd.concat([original_df["total_energy"], training_df["total_energy"]])
    y_all = pd.concat([original_df["max_load"], training_df["max_load"]])
    x_edges = np.quantile(x_all, np.linspace(0, 1, N_QUANTILE_BINS + 1))
    y_edges = np.quantile(y_all, np.linspace(0, 1, N_QUANTILE_BINS + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax in axes:
        ax.set_xlabel("total_energy")
        ax.set_ylabel("max_load")

    hist_original = axes[0].hist2d(original_df["total_energy"], original_df["max_load"], bins=[x_edges, y_edges], cmap="Blues")
    axes[0].set_title("Original energy.db distribution")
    fig.colorbar(hist_original[3], ax=axes[0], label="count")

    hist_training = axes[1].hist2d(training_df["total_energy"], training_df["max_load"], bins=[x_edges, y_edges], cmap="Greens")
    axes[1].set_title(f"Training CH pool (n={len(training_df)})")
    fig.colorbar(hist_training[3], ax=axes[1], label="count")

    out_path = OUTPUT_DIR / "training_vs_original_2d_hist.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved training-vs-original 2D histogram: {out_path}")


def main() -> None:
    original_df = load_original_metadata()
    ch_pool_df = load_ch_pool_metadata()
    preferred_ids = load_preferred_player_ids()

    original_bin_edges = compute_bin_edges(original_df)
    original_binned = add_bin_columns(original_df, original_bin_edges)
    ch_binned = add_bin_columns(ch_pool_df, original_bin_edges)

    original_dist = build_distribution(original_binned)
    runtime_sample = sample_runtime_profiles(ch_binned, original_dist, preferred_ids)
    save_runtime_sample(runtime_sample)

    all_player_ids = set(ch_pool_df["player_id"].astype(str).unique())
    runtime_player_ids = set(runtime_sample["player_id"].astype(str).unique())
    remaining_ids = all_player_ids - runtime_player_ids
    training_df = ch_pool_df[ch_pool_df["player_id"].astype(str).isin(remaining_ids)].copy()
    save_training_sample(training_df)

    summary = summarize(runtime_sample)
    summary_path = OUTPUT_DIR / "runtime_sample_summary.csv"
    summary.to_csv(summary_path)

    print(f"Runtime sample rows: {len(runtime_sample)}")
    print(f"Unique runtime player IDs: {len(runtime_player_ids)}")
    print(f"Unique IDs left in CH pool: {len(remaining_ids)}")
    print(f"Training rows from remaining IDs: {len(training_df)}")
    print(f"Sample saved to: {CH_DB_PATH}")
    print(f"Summary saved to: {summary_path}")
    print(summary)

    plot_runtime_vs_original(original_df, runtime_sample)
    plot_training_vs_original(original_df, training_df)


if __name__ == "__main__":
    main()
