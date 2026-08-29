# feature dorp report

here we explain what features we are dropping and why
to make life easy we just drop everything with gain = 0.0
and speculate on why the feature is useless

## base load

base_load_lag_12_is_pad
base_load_lag_1_is_pad
base_load_lag_8_is_pad
base_load_lag_2_is_pad
base_load_lag_4_is_pad

doesnt add new information. pad value is -1 so xgb probably just leverages that.


## pv gen

pv_lag_12_is_pad
pv_lag_1_is_pad
pv_lag_2_is_pad
pv_lag_8_is_pad
pv_lag_4_is_pad

steps_to_daylight_end
steps_to_daylight_start

"is_pad": see above
"steps_to_daylight boundary" : similar. steps to daylight are encoded elsewhere, eg time and lag features

or shorter: there are stronger alternatives with the same information, and some more, so XGB uses those instead


## ev status

steps_to_end1_latest,
end1_latest : redundant. only relevant in edge cases of phase 'driving1'. proxy: lag features probably. also it is just a transformation of timestep

steps_to_end2_latest
end2_latest: similar

end2_observed,
end1_observed: -> phase_id
end2: -> phase_id, time
observed_window_length_2 -> phase_id
max_commute_steps_2 -> phase_id

status_lag_1_is_pad
status_lag_2_is_pad
status_lag_8_is_pad
status_lag_4_is_pad -> lag = -1, no new info 
start2_observed: no new information -> phase_id
start1_observed: ""
start2_earliest: time
start1_earliest: time


max_commute_steps_1 -> learned implicitly but also contained in window length slack. also, time, lag

status_lag_8: too far in the past

steps_to_start2_earliest -> time
steps_to_start1_earliest -> time
window_length_slack_2 -> we are already back home