import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = next(
    (parent for parent in [Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[0]] if (parent / "src").exists()),
    None,
)
if REPO_ROOT is None:
    raise RuntimeError("Could not locate the repository root from the script location.")

sys.path.insert(0, str(REPO_ROOT))

from runtime_config import RuntimeConfig


PLAYER_ID = 1
RUN_ID = 5
SCENARIO = "default_scenario"
INPUT_TABLES = [
    "base_load",
    "buy_price",
    "ev1_at_charging_station",
    "ev1_at_home",
    "ev1_buy_price",
    "ev1_load",
    "ev1_max_charge",
    "ev2_at_charging_station",
    "ev2_at_home",
    "ev2_buy_price",
    "ev2_load",
    "ev2_max_charge",
    "pv_gen",
    "sell_price",
    "ev1_status",
    "ev2_status",
]
RESULT_METRICS = ["net_cost", "net_load", "total_consumption", "total_cost"]


def build_output_dirs() -> tuple[Path, Path]:
    input_dir = REPO_ROOT / "plots" / "player_1" / "input_profiles"
    result_dir = REPO_ROOT / "plots" / "player_1" / "result_profiles"
    input_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, result_dir


def load_input_series(conn: sqlite3.Connection, table_name: str, player_id: int) -> pd.DataFrame:
    query = f'SELECT period, "{player_id}" AS value FROM "{table_name}" ORDER BY period'
    df = pd.read_sql_query(query, conn)
    if df.empty:
        return df

    df = df.dropna(subset=["period", "value"]).copy()
    if df.empty:
        return df

    df["period"] = pd.to_numeric(df["period"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["period", "value"]).copy()
    df["hour"] = (df["period"] - 1) / 4.0
    return df


def save_line_plot(df: pd.DataFrame, output_path: Path, title: str, y_label: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if df.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(df["hour"], df["value"], linewidth=1.8)
        ax.set_xlabel("Hour")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_scalar_plot(value: float | None, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if value is None:
        ax.text(0.5, 0.5, "No result available", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.bar([0], [value], color="tab:blue")
        ax.set_xticks([0])
        ax.set_xticklabels([title])
        ax.set_ylabel("Value")
        ax.grid(axis="y", alpha=0.25)
        ax.text(0, value, f"{value:.3f}", ha="center", va="bottom")

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def load_result_metric(conn: sqlite3.Connection, metric: str) -> float | None:
    query = f"""
        SELECT {metric} AS value
        FROM results
        WHERE run_id = ? AND player_id = ? AND scenario = ?
        ORDER BY rowid DESC
        LIMIT 1
    """
    result = pd.read_sql_query(query, conn, params=(RUN_ID, PLAYER_ID, SCENARIO))
    if result.empty:
        return None
    value = pd.to_numeric(result.iloc[0]["value"], errors="coerce")
    return None if pd.isna(value) else float(value)


def export_input_profiles(conn: sqlite3.Connection, input_dir: Path) -> None:
    for table_name in INPUT_TABLES:
        df = load_input_series(conn, table_name, PLAYER_ID)
        output_path = input_dir / f"{table_name}_player_{PLAYER_ID}.png"
        save_line_plot(df, output_path=output_path, title=f"{table_name} (player {PLAYER_ID})", y_label="Value")


def export_result_profiles(conn: sqlite3.Connection, result_dir: Path) -> None:
    for metric in RESULT_METRICS:
        value = load_result_metric(conn, metric)
        output_path = result_dir / f"{metric}_player_{PLAYER_ID}_run_{RUN_ID}.png"
        save_scalar_plot(value, output_path=output_path, title=f"{metric} (player {PLAYER_ID}, run {RUN_ID})")


def main() -> None:
    input_dir, result_dir = build_output_dirs()
    with sqlite3.connect(RuntimeConfig.SQLITE_PATH) as conn:
        export_input_profiles(conn, input_dir)
        export_result_profiles(conn, result_dir)

    print(f"Saved input profiles to {input_dir}")
    print(f"Saved result profiles to {result_dir}")


if __name__ == "__main__":
    main()
