# RandomForest vs. XGBoost: Laufzeit-Vergleich (Begründung für RF-Ausschluss)

Datum: 2026-09-10
Script: [`training/benchmarks/rf_vs_xgb_latency.py`](../training/benchmarks/rf_vs_xgb_latency.py)

## Fragestellung

Lohnt es sich, RandomForest als Modellfamilie neben XGBoost weiter zu pflegen? Setup:
100 Trees, vergleichbare Hyperparameter (`max_depth=6`, `n_jobs=1`), 18 Features
(entspricht dem `base_load`-Feature-Set aus `model_config.py`).

## Ergebnis: Batch vs. Single-Row

| | Fit-Zeit | Batch predict/Zeile | Single-row predict |
|---|---|---|---|
| RandomForest (100 trees) | 11.6 s | 10.4 µs | **7 419 µs** |
| XGBoost (100 trees) | 0.9 s | 1.95 µs | **362 µs** |

**XGBoost ist beim Training ~13x, im Batch-Predict ~5x und bei Einzelzeilen-Predict
~20x schneller.** Der Single-Row-Fall ist der relevante, weil die MPC-Predictoren
Modelle immer mit einer Zeile pro Aufruf ansprechen (kein Batching über Zeit oder
Haushalte).

## Warum trifft das die Simulation konkret?

Beide Predictor-Architekturen im Repo rufen `model.predict()` pro Haushalt einzeln auf
(bestätigt in [`mpc_controller.py`](../src/simulation/controllers/mpc/mpc_controller.py),
`predictor.predict(household, planning_horizon)` — keine Batch-Vektorisierung über
Haushalte):

- **recursive Predictor** (z.B. [`recursive/helpers/base_load.py`](../src/simulation/controllers/mpc/predictors/ml/recursive/helpers/base_load.py)):
  ruft ein Modell rekursiv `horizon - 1` mal auf (jeder Schritt hängt vom vorigen ab).
  Bei `horizon=96`, 96 Replanning-Schritten/Tag, 4 Targets (base_load, pv_gen,
  ev1_status, ev2_status) ergibt das **36 480 Einzelzeilen-Calls pro Haushalt/Tag**.
- **composite Predictor** (z.B. [`composite/helpers/base_load.py`](../src/simulation/controllers/mpc/predictors/ml/composite/helpers/base_load.py)):
  hält pro Target eine `model_bank` mit einem Modell pro direktem Horizont-Offset
  (z.B. 8 Modelle) und interpoliert dazwischen statt zu rekursieren. Das ergibt nur
  **3 072 Calls pro Haushalt/Tag** (~12x weniger als recursive).

### Hochrechnung (Einzelzeilen-Latenz × Calls × Haushalte)

| Szenario | Calls/Haushalt/Tag | RF, 200 Haushalte | XGB, 200 Haushalte |
|---|---|---|---|
| recursive | 36 480 | ~54 100 s (~15 h) | ~2 640 s (~44 min) |
| composite | 3 072 | ~4 560 s (~76 min) | ~220 s (~4 min) |

Der 20x-Faktor bleibt in beiden Architekturen gleich (er kommt aus der Latenz pro
Aufruf, nicht aus der Aufrufzahl) — nur die absolute Größenordnung unterscheidet sich.
Selbst im günstigeren composite-Fall kostet RF bei 200 Haushalten über eine Stunde
allein für Prädiktion, XGBoost unter 4 Minuten.

## Warum ist RF pro Aufruf so viel langsamer? (empirisch belegt, nicht "unkompiliert")

Beide Modellfamilien nutzen kompilierten Code für die eigentliche Baumtraversierung
(sklearn: Cython, XGBoost: C++). Der Unterschied liegt **nicht** darin, dass RF "erst
noch kompiliert werden muss" — cProfile auf 2000 Single-Row-Calls eines RF mit 100
Trees zeigt, wo die Zeit tatsächlich hingeht:

```
200000 calls -> joblib Parallel-Dispatch pro Baum       36.5 s kumulativ
1000000 calls -> warnings.filterwarnings (!)              5.8 s
408000  calls -> __sklearn_tags__ / get_tags              5.8 s
204000  calls -> check_is_fitted                           4.6 s
200000  calls -> sklearn tree_.predict (echte Baumarbeit)  10.8 s  <- einziger "echter" Anteil
```

sklearns `RandomForestRegressor.predict()` dispatcht für **jeden der 100 Bäume
einzeln** durch `joblib.Parallel` (Thread-Dispatch-Overhead auch bei `n_jobs=1`) und
zahlt bei jedem Aufruf generische Validierungs-/Kompatibilitäts-Kosten
(`check_is_fitted`, `__sklearn_tags__`, `warnings.filterwarnings`). Das passiert bei
jedem einzelnen `.predict()`-Call neu, unabhängig von der Eingabegröße.

XGBoost hält alle Bäume in einer flachen C++-Struktur und durchläuft sie in **einem**
nativen Call ohne Python-Zwischenschritte pro Baum und ohne Thread-Pool-Dispatch.

## Ist der Overhead vermeidbar?

Teilweise, aber nicht mit sklearn-Bordmitteln:

| Variante | Latenz/Call | Faktor vs. XGB |
|---|---|---|
| `RandomForestRegressor.predict()` (Standard) | 7 419 µs | 20x langsamer |
| Rohe `tree_.predict()` je Baum + manuelles Mitteln (Joblib/Validierung umgangen) | 694 µs | ~1.9x langsamer |
| XGBoost `.predict()` | 362 µs | — |

Das Umgehen des offiziellen `predict()`-Pfads (private `tree_`-Attribute direkt
ansprechen) bringt ~10x, ist aber unsupported/fragil (bricht potenziell bei
sklearn-Updates), selbst zu pflegen und bleibt trotzdem ~2x langsamer als natives
XGBoost.

Alternativen, die das Problem sauberer lösen würden (nicht getestet/installiert):
- **Intel `scikit-learn-intelex`**: patcht sklearn mit oneDAL-Backend.
- **Treelite**: kompiliert einen trainierten RF in echten nativen Code.
- **Haushalts-Batching pro Zeitschritt**: alle Haushalte replanen synchron pro
  globalem Timestep — der erste rekursive Schritt aller Haushalte könnte in einem
  Batch-Call gebündelt werden. Würde beiden Modellfamilien helfen, erfordert aber
  einen Architektur-Umbau und hilft nur pro Horizont-Schritt, nicht über den ganzen
  rekursiven Horizont hinweg.

## Zusätzliche Punkte (nicht-Laufzeit-bezogen)

- **Modellgüte:** Eigene RMSE-Experimente (single-step prediction) zeigen keinen
  nennenswerten Unterschied zwischen RF und XGBoost. Die Entscheidung für XGBoost
  kostet also keine Prognosegüte, sie ist rein laufzeitgetrieben.
- **Feature-Engineering-Aufwand:** XGBoost übernimmt Feature-Selektion faktisch
  selbst (irrelevante Features werden über die Splits kaum genutzt und tragen
  praktisch nicht zum Gain bei). Dadurch kann man großzügig viele Lags/Rolling-Stats
  etc. als Kandidaten-Features reinwerfen, ohne pro Horizont-Modell manuell zu
  kuratieren — RF verlangt hier tendenziell mehr manuelles Pruning, um nicht an
  irrelevanten Splits zu verlieren.

## Fazit

RandomForest raus, XGBoost als einzige Modellfamilie behalten:
- 13x schnelleres Training, 5x schnellerer Batch-Predict, 20x schnellerer
  Single-Row-Predict bei vergleichbaren Hyperparametern.
- Der Single-Row-Fall ist exakt das Zugriffsmuster der rekursiven MPC-Prädiktion.
- Jede Mitigation für RF (Bypass-Hacks, Fremd-Libraries, Architektur-Umbau) bedeutet
  zusätzliche Komplexität/Abhängigkeiten, die XGBoost von Haus aus nicht braucht.
- Kein Trade-off bei der Modellgüte (RMSE ~gleich) und weniger Feature-Engineering-
  Aufwand on top.
