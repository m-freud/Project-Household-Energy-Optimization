from temperature_profiles import generate_temperature_profile, add_noise
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


# guess what, we just do a for loop.
T_day_profile_predicted = generate_temperature_profile(time_steps=100, amplitude=5, offset=15)
T_day_profile_real = add_noise(T_day_profile_predicted, noise_level=1)


def get_temperature_prediction(horizon):
    return T_day_profile_predicted[horizon]

heat_transfer_coefficient = 0.1  # how much the outside temperature affects the room temperature
alpha = heat_transfer_coefficient

T_room = 20
T_outside = 15
T_control_action = 0
beta = 1.0  # control gain

constraints = {
    'T_room_min': 18,
    'T_room_max': 22,
    'T_control_action_min': -1,
    'T_control_action_max': 1
}

horizon_len = 10
room_temperature_history = [T_room]
outside_temperature_history = []
control_history = []

for i in range(0, len(T_day_profile_real) - horizon_len):
    # 1. predict outside temperature for the MPC horizon
    horizon = slice(i, i + horizon_len)
    T_profile_predicted = get_temperature_prediction(horizon)

    # 2. update the current room temperature with real outside temperature and last action
    T_outside = T_day_profile_real[i]
    outside_temperature_history.append(T_outside)
    T_room = T_room + alpha * (T_outside - T_room) + beta * T_control_action

    # 3. optimize control actions for the horizon and apply only the first action
    u = cp.Variable(horizon_len)
    T = cp.Variable(horizon_len + 1)

    mpc_constraints = [T[0] == T_room]
    for k in range(horizon_len):
        mpc_constraints += [
            T[k + 1] == T[k] + alpha * (T_profile_predicted[k] - T[k]) + beta * u[k],
            T[k + 1] >= constraints['T_room_min'],
            T[k + 1] <= constraints['T_room_max'],
            u[k] >= constraints['T_control_action_min'],
            u[k] <= constraints['T_control_action_max'],
        ]

    objective = cp.Minimize(cp.sum_squares(u))
    problem = cp.Problem(objective, mpc_constraints)
    problem.solve(solver=cp.OSQP, warm_start=True)

    if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and u.value is not None:
        T_control_action = float(u.value[0])
    else:
        # If infeasible, hold last action and keep it within physical bounds.
        T_control_action = float(np.clip(
            T_control_action,
            constraints['T_control_action_min'],
            constraints['T_control_action_max'],
        ))

    room_temperature_history.append(T_room)
    control_history.append(T_control_action)

print('Final room temperature:', round(room_temperature_history[-1], 3))
print('First 10 control actions:', np.round(control_history[:10], 3))

# Plot results to quickly inspect controller behavior.
time_axis = np.arange(len(control_history))

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(time_axis, room_temperature_history[:-1], label='Room temperature', color='tab:blue')
axes[0].plot(time_axis, outside_temperature_history, label='Outside temperature (real)', color='tab:orange', alpha=0.8)
axes[0].axhline(constraints['T_room_min'], color='tab:red', linestyle='--', linewidth=1, label='T min/max')
axes[0].axhline(constraints['T_room_max'], color='tab:red', linestyle='--', linewidth=1)
axes[0].set_ylabel('Temperature')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)

axes[1].step(time_axis, control_history, where='post', color='tab:green', label='Control action')
axes[1].axhline(constraints['T_control_action_min'], color='tab:purple', linestyle='--', linewidth=1, label='u min/max')
axes[1].axhline(constraints['T_control_action_max'], color='tab:purple', linestyle='--', linewidth=1)
axes[1].set_ylabel('Control u')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)

axes[2].plot(time_axis, np.square(control_history), color='tab:brown', label='u^2 (stage cost)')
axes[2].set_ylabel('Cost term')
axes[2].set_xlabel('Timestep')
axes[2].legend(loc='best')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = '_MPC_INTRO/mpc_run_plot.png'
plt.savefig(plot_path, dpi=120)
print('Saved plot to:', plot_path)
plt.show()
