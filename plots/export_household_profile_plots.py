import argparse
import sqlite3
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


repo_root = next((p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents] if (p / "src").exists()), None)
if repo_root is None:
    raise RuntimeError("Could not locate repository root containing 'src'.")
sys.path.insert(0, str(repo_root))

from runtime_config import RuntimeConfig


def _load_wide_table(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY period", conn)


def _household_columns(df: pd.DataFrame) -> list[str]:
    return sorted([column for column in df.columns if str(column).isdigit()], key=int)


def _ensure_dirs(base_dir: Path) -> dict[str, Path]:
    output_dirs = {
        "base_load": base_dir / "base_load",
        "pv_gen": base_dir / "pv_gen",
        "ev_status": base_dir / "ev_status",
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return output_dirs


def _plot_single_series(periods: pd.Series, values: pd.Series, title: str, y_label: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(periods, values, linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_ev_status(periods: pd.Series, ev1_status: pd.Series, ev2_status: pd.Series, output_path: Path, household_id: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(periods, ev1_status, linewidth=1.6, drawstyle="steps-post")
    axes[0].set_title(f"Household {household_id} - EV1 Status")
    axes[0].set_ylabel("EV1")
    axes[0].set_yticks([0, 1, 2])
    axes[0].grid(alpha=0.25)

    axes[1].plot(periods, ev2_status, linewidth=1.6, drawstyle="steps-post")
    axes[1].set_title(f"Household {household_id} - EV2 Status")
    axes[1].set_ylabel("EV2")
    axes[1].set_xlabel("Timestep")
    axes[1].set_yticks([0, 1, 2])
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def export_household_plots(limit: int | None = None) -> None:
    plots_dir = repo_root / "plots"
    output_dirs = _ensure_dirs(plots_dir)

    with sqlite3.connect(RuntimeConfig.SQLITE_PATH) as conn:
        base_load = _load_wide_table(conn, "base_load")
        pv_gen = _load_wide_table(conn, "pv_gen")
        ev1_home = _load_wide_table(conn, "ev1_at_home")
        ev1_station = _load_wide_table(conn, "ev1_at_charging_station")
        ev2_home = _load_wide_table(conn, "ev2_at_home")
        ev2_station = _load_wide_table(conn, "ev2_at_charging_station")

    candidate_sets = [
        set(_household_columns(base_load)),
        set(_household_columns(pv_gen)),
        set(_household_columns(ev1_home)),
        set(_household_columns(ev1_station)),
        set(_household_columns(ev2_home)),
        set(_household_columns(ev2_station)),
    ]
    common_households = sorted(set.intersection(*candidate_sets), key=int)

    if limit is not None:
        common_households = common_households[:limit]

    periods = pd.to_numeric(base_load["period"], errors="coerce").fillna(0).astype(int)

    for household_col in common_households:
        household_id = int(household_col)
        suffix = f"household_{household_id:03d}.png"

        base_values = pd.to_numeric(base_load[household_col], errors="coerce").fillna(0.0)
        pv_values = pd.to_numeric(pv_gen[household_col], errors="coerce").fillna(0.0)

        ev1_status = 1 - pd.to_numeric(ev1_home[household_col], errors="coerce").fillna(0).astype(int) + pd.to_numeric(ev1_station[household_col], errors="coerce").fillna(0).astype(int)
        ev2_status = 1 - pd.to_numeric(ev2_home[household_col], errors="coerce").fillna(0).astype(int) + pd.to_numeric(ev2_station[household_col], errors="coerce").fillna(0).astype(int)

        _plot_single_series(
            periods=periods,
            values=base_values,
            title=f"Household {household_id} - Base Load",
            y_label="kW",
            output_path=output_dirs["base_load"] / suffix,
        )

        _plot_single_series(
            periods=periods,
            values=pv_values,
            title=f"Household {household_id} - PV Generation",
            y_label="kW",
            output_path=output_dirs["pv_gen"] / suffix,
        )

        _plot_ev_status(
            periods=periods,
            ev1_status=ev1_status,
            ev2_status=ev2_status,
            output_path=output_dirs["ev_status"] / suffix,
            household_id=household_id,
        )

    print(f"Exported plots for {len(common_households)} households.")
    print(f"Output folders:\n- {output_dirs['base_load']}\n- {output_dirs['pv_gen']}\n- {output_dirs['ev_status']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export household profile plots for manual duplicate inspection.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of households to export.")
    args = parser.parse_args()

    export_household_plots(limit=args.limit)


if __name__ == "__main__":
    main()
