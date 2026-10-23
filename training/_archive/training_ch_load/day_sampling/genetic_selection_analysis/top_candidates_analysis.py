from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = (
	Path(__file__).parent.parent
	/ "genetic_selection_results"
	/ "rmse_evolution_n_days_150_pop_150_gen_42_mut_0.2_seed_42_top_sets.csv"
)


def _parse_day_sets(value: str) -> pd.DataFrame:
	rows = []
	for set_number, serialized_set in enumerate(value.splitlines(), start=1):
		for entry in serialized_set.split(";"):
			household_id, date = entry.rsplit("_", maxsplit=1)
			rows.append(
				{
					"set_number": set_number,
					"household_id": household_id,
					"date": pd.to_datetime(date),
				}
			)
	return pd.DataFrame(rows)


def _load_selected_days(input_path: Path) -> pd.DataFrame:
	candidates = pd.read_csv(input_path)
	if "day_set" not in candidates.columns:
		raise ValueError(f"Missing 'day_set' column in {input_path}")
	return _parse_day_sets("\n".join(candidates["day_set"].dropna().astype(str)))


def _save_count_plot(
	counts: pd.Series,
	output_path: Path,
	title: str,
	x_label: str,
) -> None:
	fig_width = max(10, len(counts) * 0.28)
	fig, ax = plt.subplots(figsize=(fig_width, 6))
	counts.plot(kind="bar", ax=ax, color="#2f6f95")
	ax.set_title(title)
	ax.set_xlabel(x_label)
	ax.set_ylabel("count")
	ax.grid(axis="y", alpha=0.25)
	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def analyze(input_path: Path, output_dir: Path) -> None:
	selected_days = _load_selected_days(input_path)
	output_dir.mkdir(parents=True, exist_ok=True)

	month_counts = (
		selected_days.assign(month=selected_days["date"].dt.to_period("M").astype(str))
		.groupby("month")
		.size()
		.rename("count")
	)
	player_counts = (
		selected_days.groupby("household_id")
		.size()
		.sort_values(ascending=False)
		.rename("count")
	)

	month_counts.to_csv(output_dir / "top_sets_month_counts.csv", header=True)
	player_counts.to_csv(output_dir / "top_sets_player_counts.csv", header=True)
	_save_count_plot(
		month_counts,
		output_dir / "top_sets_month_counts.png",
		"Selected day count by month",
		"month",
	)
	_save_count_plot(
		player_counts,
		output_dir / "top_sets_player_counts.png",
		"Selected day count by player",
		"player household_id",
	)

	print(f"Input: {input_path}")
	print(f"Top sets: {selected_days['set_number'].nunique()}")
	print(f"Selected household-days: {len(selected_days)}")
	print(f"Saved month counts and plot to: {output_dir}")
	print(f"Saved player counts and plot to: {output_dir}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Analyze household-day distribution in top GA sets.")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
	parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
	args = parser.parse_args()
	analyze(args.input, args.output_dir)


if __name__ == "__main__":
	main()
