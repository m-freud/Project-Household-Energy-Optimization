# End-to-End Code Review Checklist

Reihenfolge: Fundament → Datenzugang → Simulation-Kern → Devices → Controller/Policies → Predictors → Dashboard

---

## 1) Konfiguration & Infrastruktur

- [x] `src/config.py`
- [x] `src/sqlite_connection.py`

---

## 2) Datenzugang & Ingestion

- [x] `src/ingestion/table_config.py`
- [x] `src/ingestion/table_loading.py`

---

## 3) Szenarios

- [x] `src/simulation/scenarios/scenario.py`
- [x] `src/simulation/scenarios/legacy_scenarios.py`

---

## 4) Simulation-Kern

- [x] `src/simulation/run_context.py`
- [x] `src/simulation/household.py`
- [x] `src/simulation/simulation.py`

---

## 5) Devices

- [x] `src/simulation/devices/bess.py`
- [x] `src/simulation/devices/ev.py`
- [x] `src/simulation/devices/pv.py`

---

## 6) Controller-Basis

- [x] `src/simulation/controllers/base_controller.py`
- [x] `src/simulation/controllers/function_controller.py`

---

## 7) Policies

- [x] `src/simulation/controllers/policies/waterfall/waterfall.py`
- [x] `src/simulation/controllers/policies/linear/linear.py`
  - Gleichmäßige Verteilung auf verbleibende Timesteps korrekt
- [x] `src/simulation/controllers/policies/linear/price_aware_linear.py`
  - Preisgewichtung und Deadline-Logik konsistent mit `linear.py`

---

## 8) MPC-Controller & Solver

- [ ] `src/simulation/controllers/mpc/mpc_controller.py`
  - `make_mpc_controller` Factory: Parameter korrekt durchgereicht
  - Horizon-Handling: was passiert wenn horizon > verbleibende Timesteps?
  - Fallback bei CVXPY-Solve-Fehler (infeasible/timeout)
- [ ] `src/simulation/controllers/mpc/config/device_buffer_config.py`
  - Buffer-Werte plausibel, konsistent mit BESS/EV-Caps in `config.py`

---

## 9) Predictor-Basis & Legacy

- [ ] `src/simulation/controllers/mpc/predictors/base_predictor.py`
  - Interface klar, Default-Implementierungen sinnvoll
- [ ] `src/simulation/controllers/mpc/predictors/oracle_predictor.py`
  - Liest tatsächliche zukünftige Werte korrekt (Zeitachse!)
- [ ] `src/simulation/controllers/mpc/predictors/moving_average_predictor.py`
  - Noch aktiv genutzt? Wenn nein: als deprecated markieren
- [ ] `src/simulation/controllers/mpc/predictors/moving_average_predictor2.py`
  - Noch aktiv genutzt? Wenn nein: als deprecated markieren

---

## 10) Hybrid-MA Predictor

- [ ] `src/simulation/controllers/mpc/predictors/hybrid_ma_predictor.py`
  - Parameter-Normalisierung (short=0 erlaubt? long >= 1?)
  - `source_average_beta`-Blending: korrekte Zeitachsen-Indizierung (`start_idx`)
- [ ] `src/simulation/controllers/mpc/predictors/hybrid_ma/moving_average.py`
  - `seed_series`: Verhalten bei leerer History korrekt
  - `persistence_alpha`: alle Modi (none / constant / linear / exponential)
  - `forecast_moving_average`: short_window=0 Fallback (Long-Only)
- [ ] `src/simulation/controllers/mpc/predictors/hybrid_ma/house_profiles.py`
  - `predict_base_load` / `predict_pv_gen`: PV-Fenster-Mask korrekt angewandt
- [ ] `src/simulation/controllers/mpc/predictors/hybrid_ma/ev_profiles.py`
  - EV-Load-Schätzung aus non-zero History sinnvoll
  - Driving-Load vs. Station/Home-Load richtig getrennt
- [ ] `src/simulation/controllers/mpc/predictors/hybrid_ma/ev_status.py`
  - Verfügbarkeits-Forecast: historische Pattern oder feste Regel?
- [ ] `src/simulation/controllers/mpc/predictors/hybrid_ma/price_profiles.py`
  - Buy/Sell-Price und EV-Buy-Price korrekt zusammengesetzt

---

## 11) Dashboard

- [ ] `src/dashboard/dashboard.py`
  - Navigation / Page-Routing vollständig
- [ ] `src/dashboard/prediction_explorer.py`
  - Default-Hyperparameter (long=96, rest=0, constant) korrekt gesetzt ✓
  - Cache-Invalidierung bei Parameteränderungen korrekt
  - Zeitachsen-Overlays (long/short/persistence window) korrekt gezeichnet
- [ ] `src/dashboard/general_performance/general_performance.py`
  - Aggregation über Haushalte / Szenarien korrekt
  - Target-Hit-Rates korrekt berechnet und angezeigt
- [ ] `src/dashboard/single_performance/single_performance.py`
  - Haushalt / Szenario / Run auswählbar
- [ ] `src/dashboard/single_performance/kpi_table.py`
  - KPIs vollständig und korrekt (net_cost, total_cost, target_met_*)
- [ ] `src/dashboard/single_performance/debug_table.py`
  - Debug-Ausgaben konsistent mit Simulation-Output
- [ ] `src/dashboard/single_performance/subplots/`
  - `plot_bess.py` – SoC-Verlauf, Charge/Discharge korrekt
  - `plot_ev.py` – EV-SoC, Availability, Max-Charge korrekt
  - `plot_pv.py` – PV-Erzeugung vs. Vorhersage
  - `plot_net_load.py` – Net Load Berechnung korrekt
  - `plot_net_cost.py` – Kostenberechnung korrekt
  - `helpers.py` – gemeinsame Hilfsfunktionen fehlerfrei

---

## 12) Kritische Querschnittsthemen (bei jedem File im Kopf behalten)

- [ ] **Zeitachse**: Ist alles 1-basiert (1..96)? Kein Mischen mit 0-basiert?
- [ ] **History-Zugriff**: Wird `timestep` oder `timestep-1` für den aktuellen Wert genutzt? Konsistent?
- [ ] **Horizon**: Wird horizon auf verbleibende Timesteps gekürzt (`96 - timestep + 1`)?
- [ ] **EV-Status/Max-Charge-Pfad**: Home-Cap vs. Station-Cap vs. None korrekt gemappt?
- [ ] **Target-Hit-Bedingung**: Schwelle, Zeitpunkt und Aggregation über alle Szenarien einheitlich?
- [ ] **DB-Schema**: Alle geschriebenen Spalten vorhanden, keine fehlenden Werte bei Pflichtfeldern?
