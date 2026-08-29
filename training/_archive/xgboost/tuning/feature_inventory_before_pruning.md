# XGBoost Feature Inventory (Before Pruning)

Snapshot date: 2026-08-12
Source of truth: `src/config.py` (`Config.XGB_FEATURES`)

This file records the current feature sets before any pruning decisions.

## Target -> Feature Set Mapping

- `base_load` uses `BASE_LOAD` (24 features)
- `pv_gen` uses `PV_GEN` (25 features)
- `ev1_status` uses `EV_STATUS` (36 features)
- `ev2_status` uses `EV_STATUS` (36 features)

## BASE_LOAD (24)

1. `timestep`
2. `base_load`
3. `n_evs_at_home`
4. `time_sin`
5. `time_cos`
6. `base_load_lag_1`
7. `base_load_lag_1_is_pad`
8. `base_load_lag_2`
9. `base_load_lag_2_is_pad`
10. `base_load_lag_4`
11. `base_load_lag_4_is_pad`
12. `base_load_lag_8`
13. `base_load_lag_8_is_pad`
14. `base_load_lag_12`
15. `base_load_lag_12_is_pad`
16. `base_load_ma_2`
17. `base_load_ma_4`
18. `base_load_ma_8`
19. `base_load_ma_16`
20. `base_load_std_4`
21. `base_load_std_8`
22. `base_load_delta_1`
23. `base_load_delta_2`
24. `base_load_accel`

## PV_GEN (25)

1. `timestep`
2. `pv_gen`
3. `time_sin`
4. `time_cos`
5. `pv_lag_1`
6. `pv_lag_1_is_pad`
7. `pv_lag_2`
8. `pv_lag_2_is_pad`
9. `pv_lag_4`
10. `pv_lag_4_is_pad`
11. `pv_lag_8`
12. `pv_lag_8_is_pad`
13. `pv_lag_12`
14. `pv_lag_12_is_pad`
15. `pv_ma_2`
16. `pv_ma_4`
17. `pv_ma_8`
18. `pv_ma_16`
19. `pv_std_4`
20. `pv_std_8`
21. `pv_delta_1`
22. `pv_delta_2`
23. `pv_accel`
24. `steps_to_daylight_start`
25. `steps_to_daylight_end`

## EV_STATUS (36)

1. `timestep`
2. `status`
3. `time_sin`
4. `time_cos`
5. `steps_in_current_state`
6. `phase_id`
7. `status_lag_1`
8. `status_lag_1_is_pad`
9. `status_lag_2`
10. `status_lag_2_is_pad`
11. `status_lag_4`
12. `status_lag_4_is_pad`
13. `status_lag_8`
14. `status_lag_8_is_pad`
15. `start1_earliest`
16. `end1_latest`
17. `start2_earliest`
18. `end2_latest`
19. `max_commute_steps_1`
20. `max_commute_steps_2`
21. `steps_to_start1_earliest`
22. `steps_to_end1_latest`
23. `steps_to_start2_earliest`
24. `steps_to_end2_latest`
25. `start1`
26. `end1`
27. `start2`
28. `end2`
29. `start1_observed`
30. `end1_observed`
31. `start2_observed`
32. `end2_observed`
33. `observed_window_length_1`
34. `observed_window_length_2`
35. `window_length_slack_1`
36. `window_length_slack_2`
