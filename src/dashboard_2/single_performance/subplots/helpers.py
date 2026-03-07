import pandas as pd
import matplotlib.pyplot as plt

def shade_ev_location_background(ax: plt.Axes, at_home_df: pd.DataFrame, at_station_df: pd.DataFrame) -> None:
	if at_home_df.empty or at_station_df.empty:
		return

	merged = at_home_df[["hour", "value"]].rename(columns={"value": "at_home"}).merge(
		at_station_df[["hour", "value"]].rename(columns={"value": "at_station"}),
		on="hour",
		how="inner",
	)

	if merged.empty:
		return

	hours = merged["hour"].tolist()
	at_home_values = merged["at_home"].tolist()
	at_station_values = merged["at_station"].tolist()

	for idx, start_hour in enumerate(hours):
		if idx < len(hours) - 1:
			end_hour = hours[idx + 1]
		else:
			end_hour = start_hour + 0.25

		if at_station_values[idx] == 1:
			color = "lightblue"
		elif at_home_values[idx] == 1:
			color = "lightgreen"
		else:
			color = "lightgrey"

		ax.axvspan(start_hour, end_hour, color=color, alpha=0.22, linewidth=0)
