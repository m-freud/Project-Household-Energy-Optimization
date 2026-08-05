# EV Status Classifier Model Features

This file lists the feature columns used by the EV status classifier in training.

Source pipeline:
- training/xgboost/features/ev_status_features.py
- training/xgboost/training/ev_status_classifier.py

Columns excluded before fit:
- next_state (target)
- household_id (identifier)
- ev_key (identifier)
- phase (string label)

Model feature columns:
- timestep
- status
- time_sin
- time_cos
- steps_in_current_state
- phase_id
- status_lag_1
- status_lag_1_is_pad
- status_lag_2
- status_lag_2_is_pad
- status_lag_4
- status_lag_4_is_pad
- status_lag_8
- status_lag_8_is_pad
- start1_earliest
- end1_latest
- start2_earliest
- end2_latest
- max_commute_steps_1
- max_commute_steps_2
- steps_to_start1_earliest
- steps_to_end1_latest
- steps_to_start2_earliest
- steps_to_end2_latest
- start1
- end1
- start2
- end2
- start1_observed
- end1_observed
- start2_observed
- end2_observed
- observed_window_length_1
- observed_window_length_2
- window_length_slack_1
- window_length_slack_2
