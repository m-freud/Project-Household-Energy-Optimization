# Project ToDo

## 1) Hybrid MA Tuning abschliessen
- [ ] Hybrid MA Tuning als abgeschlossen markieren (kein weiterer Parameter-Sweep)
- [ ] Aktuellen Stand und Entscheidung im Kopf behalten: Fokus auf robuste, einfache MPC-Variante

## 2) Umfassende Code Review
- [ ] End-to-end Review der Simulation, Predictor-Logik, MPC-Constraints und Fallbacks
- [ ] Off-by-one / Zeitachsen-Pruefung (Status, Prices, Max-Charge, Horizon)
- [ ] EV-Status/Availability und Max-Charge-Pfade auf Konsistenz pruefen
- [ ] Kritische Findings priorisieren und sauber fixen

## 3) Re-Validation mit 250x6 Run (0/96)
- [ ] Voller Run: 250 Haushalte x 6 Szenarien mit 0/96-Setup
- [ ] Target-Hit-Rates und Miss-Pattern auswerten (insb. EV2)
- [ ] Vergleich gegen letzten stabilen Stand dokumentieren

## 4) Safety Fix nur falls noetig
- [ ] Wenn nach Review/Run noch Misses auftreten: kleinen Charge-Buffer ueber Ziel setzen
- [ ] Buffer so klein wie moeglich halten (minimal invasiv)
- [ ] Nochmal kurz revalidieren, dass Misses weg sind und Kosten nicht unnoetig steigen

## 5) Gscheide Auswertung + README
- [ ] Finale Auswertung zusammenfassen (Metriken, Trade-offs, verbleibende Grenzen)
- [ ] README auf finalen technischen Stand bringen
- [ ] Klarer Storyflow: Waterfall < einfacher MPC < Oracle, mit sauberem Fazit

## 6) UI aufhuebschen (optional)
- [ ] Dashboard/Plots visuell aufraeumen und konsistenter machen
- [ ] Kleine UX-Verbesserungen, solange sie den Scope nicht aufblasen

## 7) XGBoost Pilot fuer Forecasting
- [ ] Minimalen XGBoost-Forecast-Pilot bauen (vermutlich base_load + pv_gen, bei EV wäre das eher overfitting)
- [ ] Haushalts-Split umsetzen (z. B. 100/100/50: train/val/test)
- [ ] Gegen MA-Baseline benchmarken (MAE/RMSE + Auswirkungen im MPC)
- [ ] Wenn besser als MA: als optionalen Predictor integrieren
- [ ] Kurz dokumentieren: Setup, Ergebnis, naechste Iteration
