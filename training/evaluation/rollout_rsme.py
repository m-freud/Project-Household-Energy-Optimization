'''
we already have direct next step rsme.
now we want rollout rsme for different horizons.
eg.

h_1 = direct next step error
h_2 = avg rollout error after 2 steps
h_24 = avg rollout error after 24 steps

then, we can form 2 different averages:
1. pooled rollout error: average of all rollout errors across all horizons (-> weighted by usage of each horizon)
2. horizon mean rollout error: average of rollout errors for each horizon (-> unweighted by usage of each horizon)

and then same for time of day:

t_1 = rollout error at timestep 1
t_2 = avg rollout error at timestep 2
t_24 = avg rollout error at timestep 24

and then we can form 2 more averages:
1. pooled time of day error: average of all rollout errors across all timesteps (-> weighted by usage of each timestep)
2. timestep mean rollout error: average of rollout errors for each timestep (-> unweighted by usage of each timestep)


'''


def get_rollout_error(model, day_profile: list[int], target="base_load"):
    day_len = len(day_profile)
    for t in range(1, day_len):
        for s in range(day_len - t):
            
