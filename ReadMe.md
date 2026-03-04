# Domestic Energy Optimization

Here we see what we can do with this dataset:
https://zenodo.org/records/11351017

paper:
https://www.sciencedirect.com/science/article/pii/S2352340923003372

It contains a simulated community of households with different setups of EVs, PVs, ...
Energy consumption and generation, etc is simulated for 1 day in 15 min intervals for 96 households (?)

Based on real user parameters from measurements.
So the data is typical but without anomalies and chaotic behaviour. Clean, idealized users but with individual
situations and energy profiles / behaviours.

Why is this data relevant?

It shows typical consumer behaviour, but without any autmoated energy decisions. Consumption behaviour is ignorant and not motivated by factors like:
- daytime
- current energy prices
- grid auslastung
- PV erzeugung
- Batteriestand
- erwartete EV-Abfahrtszeiten
- grid constraints
- batterie effizienz
- wetter
- verbrauchsprognose
- uvm

So we get a good model of a customer and can show real optimization

## Streamlit dashboard

For lightweight demos/presentations without Docker/Grafana:

1. Install dependencies
	- `pip install -r requirements.txt`
2. Run simulation once (to populate SQLite measurement tables)
	- `python -m src.simulation.main`
3. Launch app
	- `streamlit run src/analysis/streamlit_app.py`

In the app you can choose:
- `player_id`
- `policy`
- `scenario`
- `day`
- multiple measurement checkboxes

Views:
- `Timeseries`: multi-line day plot for selected measurements
- `Summary`: total cost and total consumption from `results`