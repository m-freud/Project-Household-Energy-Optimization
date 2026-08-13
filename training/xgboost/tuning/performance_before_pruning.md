# Performance Before Pruning

Date: 2026-08-12
Benchmark run_id: 15
Scope: test_set, all scenarios, controllers `mpc_history_avg` vs `mpc_xgb`

## XGBoost Hyperparameters Used

Best params loaded from `training/xgboost/tuning/results.csv` and used by `training/xgboost/training/train_models.py`:

- `base_load`: learning_rate=0.05, n_estimators=100, max_depth=3, mean_rmse=0.4603579228434122
- `pv_gen`: learning_rate=0.05, n_estimators=600, max_depth=5, mean_rmse=0.1146124311719463
- `ev1_status`: learning_rate=0.1, n_estimators=300, max_depth=3, mean_log_loss=0.0820305958390235
- `ev2_status`: learning_rate=0.1, n_estimators=300, max_depth=3, mean_log_loss=0.0824457556009292

## Run 15 Controller Comparison

Rows covered:

- 240 total rows
- 120 rows per policy
- 6 scenarios
- 20 households

Aggregate metrics:

| policy | pairs | avg_total_cost | avg_net_cost | avg_net_load | bess_target_rate | ev1_target_rate | ev2_target_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| mpc_history_avg | 120 | 8.5550028067421 | 7.805227806742101 | 58.71462041112573 | 1.0 | 1.0 | 1.0 |
| mpc_xgb | 120 | 8.23587881209847 | 7.486103812098469 | 54.90590321507042 | 1.0 | 1.0 | 1.0 |
| mpc_oracle | 120 | 7.912110176109991 | 7.162335176109991 | 55.226935204408015 | 1.0 | 1.0 | 1.0 |
Pairwise `total_cost` delta (`mpc_xgb - mpc_history_avg`) on matched `(scenario, player_id)` pairs:

- pairs: 120
- mean_delta: -0.31912399464363167
- median_delta: -0.24793537521085152
- xgb_wins: 120
- history_avg_wins: 0
- ties: 0

Scenario breakdown (mean delta = `mpc_xgb - mpc_history_avg`):

| scenario | pairs | mean_delta | xgb_wins | history_avg_wins | ties |
|---|---:|---:|---:|---:|---:|
| default_scenario | 20 | -0.35644627215183095 | 20 | 0 | 0 |
| relaxed_high_start | 20 | -0.3126604439090256 | 20 | 0 | 0 |
| relaxed_low_start | 20 | -0.37070456125291856 | 20 | 0 | 0 |
| stressed_high_start | 20 | -0.2547941799837807 | 20 | 0 | 0 |
| stressed_low_start | 20 | -0.34449863284219406 | 20 | 0 | 0 |
| stressed_mid_start | 20 | -0.2756398777220403 | 20 | 0 | 0 |

## Run 16 Controller Comparison: xgb vs oracle

This section compares `mpc_oracle` (run 16) against `mpc_xgb` (run 15) using the same test_set and scenario scope.

Aggregate metrics:

| policy | pairs | avg_total_cost | avg_net_cost | avg_net_load | bess_target_rate | ev1_target_rate | ev2_target_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| mpc_xgb (run 15) | 120 | 8.23587881209847 | 7.486103812098469 | 54.90590321507042 | 1.0 | 1.0 | 1.0 |
| mpc_oracle | 120 | 7.912110176109991 | 7.162335176109991 | 55.226935204408015 | 1.0 | 1.0 | 1.0 |

Scenario breakdown:

| scenario | pairs | mean_delta | xgb_wins | oracle_wins | ties |
|---|---:|---:|---:|---:|---:|
| default_scenario | 20 | -0.3440545749155711 | 0 | 20 | 0 |
| relaxed_high_start | 20 | -0.23866981763148182 | 0 | 20 | 0 |
| relaxed_low_start | 20 |-0.3913230433354176 | 0 | 20 | 0 |
| stressed_high_start | 20 | -0.29189977751237833 | 0 | 20 | 0 |
| stressed_low_start | 20 | -0.32588970533306066 | 0 | 20 | 0 |
| stressed_mid_start | 20 | -0.3507748972029606 | 0 | 20 | 0 |


## Summary

Before feature pruning, `mpc_xgb` outperformed `mpc_history_avg` on total cost in every matched pair in run 15 while preserving full target compliance (all target rates = 1.0 for both controllers).
