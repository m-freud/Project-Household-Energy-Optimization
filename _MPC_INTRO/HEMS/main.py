import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from pathlib import Path


# gradually build a hems mpc

# V1:
# BESS -> charge/discharge action
# end goal -> constraint
# price profile -> minimize price

# later: add base load (human)



price_profile_1 = [2] * 33 + [1] * 33 + [3] * 34

horizon_len = 100

constraints = {
    'targets' : [(len(price_profile_1), 0.8)], # targets are on global SoC state index
    'bess_charge_min' : 0, # charge-only mock (no discharge/export path yet)
    'bess_charge_max' : 1,
    'bess_soc_min' : 0,
    'bess_soc_max' : 1,
}

bess_soc = 0.3
bess_charge_action = 0.0
charge_coefficient = 0.1

net_cost = 0.0
cost_history = []
soc_history = [bess_soc]
action_history = []
cumulative_cost_history = []


for i in range(len(price_profile_1)):
    # 1. predict
    pred_horizon = min(horizon_len, len(price_profile_1) - i)
    pred_price_profile = price_profile_1[i:i + pred_horizon]

    # 2. update bess soc and cost history with last action
    bess_charge_action = bess_charge_action if bess_charge_action is not None else 0.0
    bess_charge = charge_coefficient * bess_charge_action
    bess_soc += bess_charge
    bess_soc = np.clip(bess_soc, constraints['bess_soc_min'], constraints['bess_soc_max'])
    cost_history.append(bess_charge * price_profile_1[i])

    # 3. optimize control actions for the horizon and apply only the first action
    bess_charge_var = cp.Variable(pred_horizon)
    bess_soc_var = cp.Variable(pred_horizon + 1)

    mpc_constraints = [bess_soc_var[0] == bess_soc]
    for target_state_index, target_soc in constraints['targets']:
        # Convert global target index to local horizon index at time step i.
        if i <= target_state_index <= i + pred_horizon:
            local_idx = target_state_index - i
            mpc_constraints += [bess_soc_var[local_idx] == target_soc]

    for t in range(pred_horizon):
        mpc_constraints += [
            bess_soc_var[t + 1] == bess_soc_var[t] + charge_coefficient * bess_charge_var[t],
            bess_charge_var[t] >= constraints['bess_charge_min'],
            bess_charge_var[t] <= constraints['bess_charge_max'],
        ]
    mpc_constraints += [
        bess_soc_var >= constraints['bess_soc_min'],
        bess_soc_var <= constraints['bess_soc_max'],
    ]

    objective = cp.Minimize(cp.sum(cp.multiply(pred_price_profile, bess_charge_var)))
    problem = cp.Problem(objective, mpc_constraints)
    problem.solve()

    bess_charge_action = bess_charge_var.value[0] if bess_charge_var.value is not None else 0.0
    action_history.append(bess_charge_action)
    soc_history.append(bess_soc)
    net_cost += cost_history[-1]
    cumulative_cost_history.append(net_cost)


time_steps = np.arange(len(price_profile_1))

fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

axes[0].plot(time_steps, price_profile_1, color='tab:purple')
axes[0].set_ylabel('Price')
axes[0].set_title('HEMS Mock MPC Debug View')
axes[0].grid(alpha=0.3)

axes[1].step(time_steps, action_history, where='post', color='tab:orange')
axes[1].set_ylabel('Action')
axes[1].grid(alpha=0.3)

axes[2].plot(np.arange(len(soc_history)), soc_history, color='tab:green')
axes[2].set_ylabel('SoC')
axes[2].set_ylim(-0.05, 1.05)
axes[2].grid(alpha=0.3)

axes[3].plot(time_steps, cumulative_cost_history, color='tab:blue')
axes[3].set_ylabel('Cum Cost')
axes[3].set_xlabel('Time Step')
axes[3].grid(alpha=0.3)

plt.tight_layout()
plot_path = Path(__file__).with_name('hems_mock_plot.png')
plt.savefig(plot_path, dpi=150)

if 'agg' in plt.get_backend().lower():
    print(f'Saved plot to: {plot_path}')
else:
    plt.show()
