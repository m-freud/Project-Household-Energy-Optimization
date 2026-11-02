# MPC-Vergleich: Run 3

## Kurzfazit

- Im `02_default_benchmark` hat `mpc_oracle` die niedrigsten mittleren Kosten. Das ist der Oracle-Referenzcontroller, kein praktisch verfügbarer ML-Predictor.
- Unter den ML-Predictors hat `mpc_single_step` im Default-Szenario die niedrigsten mittleren Kosten: `11.940` je Haushalt. Es liegt vor `mpc_multi_step_recursive` (`12.034`), `mpc_multi_step_high_res` (`12.123`) und `mpc_multi_model_high_res` (`12.243`).
- `mpc_single_step` gewinnt den direkten Kostenvergleich gegen `mpc_multi_step_recursive` bei 193 von 250 Haushalten; der Recursive-Controller gewinnt bei 56, ein Haushalt liegt innerhalb der Gleichstandstoleranz.
- Die High-res-Varianten verbessern ihre jeweiligen linearen Vorgänger im Mittel leicht, schlagen `mpc_single_step` aber nicht.
- BESS- und EV1-Ziele werden im Default-Szenario von allen Controllern bei allen 250 Haushalten erfüllt. EV2 ist schwieriger: Die Erfüllung aller EV2-Ziele liegt je Controller zwischen 96.8 % und 99.6 %.
- Die Hypothese zu weniger Trainingsdaten an späten Horizonten ist durch die Parquets belegt: Bei Load sinkt die Zahl gültiger Samples von 95,000 für `y1` auf 1,000 für `y95`; bei PV von 51,775 auf 545. Das ist ein plausibler Einflussfaktor, aber kein Beweis, dass er allein den Kostenunterschied verursacht.

## Datenbasis und Vergleichbarkeit

Ausgewertet wurden alle Zeilen mit `run_id = '3'` in `sqlite/results.db`. Jede aufgeführte Controller-Szenario-Kombination hat Ergebnisse für 250 unterschiedliche Haushalte.

| Szenario | In Run 3 gespeicherte MPC-Controller | Haushalte je Controller |
|---|---:|---:|
| `00_baseline` | 5 | 250 |
| `02_default_benchmark` (`default_scenario`) | 9 | 250 |
| `22_early_stress` | 5 | 250 |

Die beiden High-res-Controller wurden nur im Default-Szenario ausgeführt. Baseline und Early-Stress eignen sich deshalb nicht für einen Vergleich aller neun Controller. `mpc_composite_horizons` ist der in den älteren Run-3-Ergebnissen gespeicherte Name des Multi-model-Vorgängers; er wird hier nicht mit `mpc_multi_model_high_res` gleichgesetzt.

Die Rang- und Siegstatistik vergleicht pro Szenario die Haushalte, für die alle dort aufgeführten Controller Ergebnisse haben. Niedrigere `net_cost`-Werte sind besser. Kostenunterschiede bis `1e-9` werden als Gleichstand behandelt; Gleichstände teilen den Sieganteil. Die Rangfolge verwendet bei Gleichständen mittlere Ränge. `total_cost` ergibt dieselbe Rangfolge: Der Unterschied `total_cost - net_cost` ist pro Haushalt controllerunabhängig.

## Default-Szenario: alle neun Controller

Mittlere und mediane Kosten beziehen sich auf die 250 Haushalte. „Sieganteil“ ist die Zahl der Haushaltssiege, wobei Gleichstände anteilig gezählt werden.

| Controller | Ø `net_cost` | Median `net_cost` | Ø Rang (1 = bester) | Sieganteil |
|---|---:|---:|---:|---:|
| `mpc_oracle` | 11.168 | 10.749 | 1.042 | 246.5 |
| `mpc_single_step` | 11.940 | 11.444 | 3.102 | 3.5 |
| `mpc_multi_step_recursive` | 12.034 | 11.565 | 4.170 | 0 |
| `mpc_multi_step_high_res` | 12.123 | 11.725 | 5.306 | 0 |
| `mpc_multi_step` | 12.177 | 11.749 | 5.842 | 0 |
| `mpc_multi_model_high_res` | 12.243 | 11.820 | 6.486 | 0 |
| `mpc_multi_model_recursive` | 12.451 | 11.893 | 5.806 | 0 |
| `mpc_composite_horizons` | 12.491 | 12.020 | 6.406 | 0 |
| `mpc_history_avg` | 12.522 | 12.261 | 6.840 | 0 |

### Direkter Vergleich mit `mpc_single_step`

Positive mittlere Differenzen in der Tabelle bedeuten, dass der jeweilige Vergleichscontroller höhere Kosten als `mpc_single_step` hatte.

| Vergleichscontroller | Ø Mehrkosten ggü. Single-step | Haushalte: Single-step günstiger | Vergleichscontroller günstiger | Gleichstand |
|---|---:|---:|---:|---:|
| `mpc_multi_step_recursive` | 0.094 | 193 | 56 | 1 |
| `mpc_multi_step_high_res` | 0.183 | 215 | 34 | 1 |
| `mpc_multi_model_high_res` | 0.303 | 221 | 28 | 1 |
| `mpc_multi_step` | 0.237 | 233 | 16 | 1 |
| `mpc_oracle` | -0.772 | 4 | 246 | 0 |

Der Vergleich der Auflösungen zeigt kleine Verbesserungen durch High-res:

- `mpc_multi_step_high_res` liegt im Mittel `0.054` unter dem linearen `mpc_multi_step`; es ist für 136 Haushalte günstiger, für 110 teurer, vier sind gleichauf.
- `mpc_multi_model_high_res` liegt im Mittel `0.248` unter `mpc_composite_horizons`; es ist für 119 Haushalte günstiger, für 130 teurer, einer ist gleichauf.
- Trotzdem bleiben beide High-res-Varianten beim Mittelwert hinter `mpc_single_step` und `mpc_multi_step_recursive`.

## Kosten und Siege in den weiteren Szenarien

Diese beiden Szenarien enthalten jeweils dieselben fünf Legacy-Controller. Die Sieganteile sind wieder anteilig bei Gleichständen.

| Szenario | Controller | Ø `net_cost` | Ø Rang | Sieganteil |
|---|---|---:|---:|---:|
| `00_baseline` | `mpc_oracle` | 2.372 | 2.042 | 160.3 |
|  | `mpc_single_step` | 2.547 | 2.620 | 25.7 |
|  | `mpc_composite_horizons` | 2.580 | 3.012 | 27.2 |
|  | `mpc_multi_step` | 2.583 | 3.104 | 26.7 |
|  | `mpc_history_avg` | 2.820 | 4.222 | 10.0 |
| `22_early_stress` | `mpc_oracle` | 11.221 | 1.070 | 241.5 |
|  | `mpc_single_step` | 11.943 | 2.458 | 1.5 |
|  | `mpc_composite_horizons` | 12.177 | 3.632 | 0 |
|  | `mpc_multi_step` | 12.345 | 3.884 | 0 |
|  | `mpc_history_avg` | 12.389 | 3.956 | 7.0 |

In beiden Szenarien liegt `mpc_single_step` im Mittel vor allen nicht-oracle Legacy-Controllern. Die High-res-Controller wurden dort nicht mit ausgeführt.

## Erfüllung der SOC-Ziele

`Final erfüllt` entspricht `target_met_*` (Ziel am letzten relevanten Deadline-Zeitpunkt). `Alle Deadlines erfüllt` entspricht `target_met_all_*` (alle im Szenario definierten SOC-Ziele). Prozentwerte beziehen sich auf 250 Haushalte; Ziele für nicht vorhandene Geräte gelten gemäß Simulationslogik als erfüllt.

| Szenario | Controller | BESS: alle | EV1: alle | EV2: alle | EV2: final |
|---|---|---:|---:|---:|---:|
| `02_default_benchmark` | `mpc_oracle` | 100 % | 100 % | 100 % | 100 % |
|  | `mpc_single_step` | 100 % | 100 % | 99.2 % | 99.6 % |
|  | `mpc_multi_step_recursive` | 100 % | 100 % | 99.2 % | 99.6 % |
|  | `mpc_multi_step_high_res` | 100 % | 100 % | 98.8 % | 99.2 % |
|  | `mpc_multi_step` | 100 % | 100 % | 98.8 % | 99.2 % |
|  | `mpc_multi_model_high_res` | 100 % | 100 % | 98.8 % | 99.2 % |
|  | `mpc_multi_model_recursive` | 100 % | 100 % | 96.8 % | 98.0 % |
|  | `mpc_composite_horizons` | 100 % | 100 % | 96.8 % | 98.0 % |
|  | `mpc_history_avg` | 100 % | 100 % | 99.6 % | 99.6 % |
| `00_baseline` | alle fünf Controller | 100 % | 100 % | 100 % | 100 % |
| `22_early_stress` | `mpc_oracle` | 100 % | 100 % | 97.6 % | 100 % |
|  | `mpc_single_step` | 100 % | 100 % | 77.2 % | 99.6 % |
|  | `mpc_composite_horizons` | 100 % | 100 % | 73.2 % | 99.2 % |
|  | `mpc_multi_step` | 100 % | 100 % | 74.8 % | 99.6 % |
|  | `mpc_history_avg` | 100 % | 100 % | 92.8 % | 99.6 % |

Im Early-Stress fällt vor allem die Erfüllung **aller** EV2-Deadlines ab; das reine finale EV2-Ziel wird bei nahezu allen Haushalten erreicht. Das finale Ziel allein unterschätzt daher die verfehlten Zwischenziele.

## Trainings-Hyperparameter

Die folgenden Werte stammen aus den aktuellen Trainingsaufrufen in `models/*/train_models.py` und den dort verwendeten Trainingsfunktionen. Die Einstellungen gelten je Target für das jeweilige ML-Modell. Multi-step trainiert ein gemeinsames Modell über `y1..y95` mit `y_offset`; Multi-model trainiert separate Horizontmodelle. `mpc_oracle` und `mpc_history_avg` verwenden keine hier aufgeführten XGBoost-Modelle.

| Target | Modell / Ziel | Learning rate | Max depth | Estimators | Objective / weitere Einstellungen |
|---|---|---:|---:|---:|---|
| Load | `XGBRegressor` / Load | 0.02 | 4 | 200 | `reg:squarederror`, `random_state=42` |
| PV | `XGBRegressor` / PV | 0.05 | 3 | 100 | `reg:squarederror`, `random_state=42` |
| EV1 status | `XGBClassifier` / EV1 | 0.1 | 2 | 200 | `multi:softprob`, `num_class=3`, `mlogloss`, `random_state=42` |
| EV2 status | `XGBClassifier` / EV2 | 0.1 | 2 | 200 | `multi:softprob`, `num_class=3`, `mlogloss`, `random_state=42` |

Die Hyperparameter sind nicht pro Horizont separat getunt: Innerhalb eines Targets werden dieselben Einstellungen für alle Horizonte verwendet. Das High-res-Training überspringt das bestehende `single_step.json`; dessen exakte Trainingshistorie lässt sich aus dem gespeicherten Modell und Run 3 allein nicht rekonstruieren. XGBoost-Modell-Dateien enthalten in diesem Workflow keine vollständige Provenienz mit Trainingstime, Datenversion und Hyperparametern.

## Warum kann Single-step trotzdem besser abschneiden?

**Weniger Samples für späte Horizonte ist ein realer Effekt.** Die Feature-Erzeugung verschiebt Targets innerhalb eines Tagesprofils und füllt nicht verfügbare Zielwerte mit `-1`. Dadurch nimmt die Zahl gültiger Trainingsbeispiele mit dem Offset ab:

| Target | `y1` | `y16` | `y64` | `y95` |
|---|---:|---:|---:|---:|
| Load | 95,000 | 80,000 | 32,000 | 1,000 |
| PV | 51,775 | 43,600 | 17,440 | 545 |
| EV1 status | 35,340 | 29,760 | 11,904 | 372 |
| EV2 status | 95,000 | 80,000 | 32,000 | 1,000 |

Die Single-step-Modelle trainieren dagegen auf dem gültigen Next-step-Ziel: 95,000 Load-Beispiele, 51,775 PV-, 35,340 EV1- und 95,000 EV2-Beispiele. Das sind für Load und EV2 jeweils 95-mal so viele Beispiele wie für `y95`. Beim gepoolten Multi-step-Modell ist die Gesamtzahl über `y1..y95` zwar groß, aber lange Offsets sind deutlich seltener vertreten; bei Multi-model betrifft die geringere Samplezahl das jeweilige späte Horizontmodell direkt.

Das ist eine plausible Erklärung, aber keine alleinige Ursachenfeststellung. Weitere Punkte:

1. Der MPC bewertet nicht nur Forecast-Fehler, sondern die resultierenden Ladeentscheidungen, Kosten und SOC-Ziele. Ein Predictor mit besserem RMSE muss nicht automatisch geringere Simulationskosten liefern.
2. Recursive Single-step kann Rollout-Fehler akkumulieren, nutzt aber pro Schritt die dichte Next-step-Trainingsmenge. Multi-step reduziert den Rollout, muss dafür horizonabhängige Muster aus weniger Beispielen lernen oder beim linearen Modell zwischen wenigen Stützstellen interpolieren.
3. Die Hyperparameter sind feste, targetweite Werte und Run 3 vergleicht keine systematische Hyperparameter-Suche. Unter-/Überanpassung oder ein passenderer Modelltyp bleiben möglich, sind durch die Ergebnisdaten aber nicht nachgewiesen.
4. Höhere Auflösung hilft hier moderat gegenüber den linearen Vorgängern, reicht jedoch nicht, um Single-step im Mittel zu übertreffen.

Um die Ursachen auseinanderzuhalten, sollten als Nächstes Forecast-Metriken pro Horizont (z. B. MAE/RMSE und EV-Klassifikationsmetriken) auf einem gemeinsamen zeitlich bzw. haushaltsbezogen getrennten Testsplit berechnet werden. Dazu die Kennzahlen nach Horizont und Target getrennt auswerten und danach mit den MPC-Kosten und SOC-Verfehlungen der gleichen Haushalte verknüpfen. Erst damit lässt sich erkennen, ob der Unterschied primär von Horizon-Samplezahlen, Hyperparametern, Forecast-Rollout oder der MPC-Optimierung kommt.
