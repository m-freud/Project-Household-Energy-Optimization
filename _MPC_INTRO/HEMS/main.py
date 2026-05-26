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

# Keep LP behavior while making tie cases deterministic and smooth.
linearize_actions = True
cost_opt_tolerance = 1e-8


def is_optimal_status(status):
    return status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


def build_total_variation_linearization(action_var, pred_horizon):
    # Linearized total variation: sum |u_t - u_{t-1}|.
    delta_var = cp.Variable(pred_horizon - 1, nonneg=True)
    tv_constraints = []

    for t in range(1, pred_horizon):
        tv_constraints += [
            delta_var[t - 1] >= action_var[t] - action_var[t - 1],
            delta_var[t - 1] >= -(action_var[t] - action_var[t - 1]),
        ]

    return delta_var, tv_constraints


def build_linear_problem(primary_cost_value, action_var, pred_horizon, base_constraints, cost_expr):
    delta_var, tv_constraints = build_total_variation_linearization(action_var, pred_horizon)
    linear_constraints = list(base_constraints)
    linear_constraints += [cost_expr <= primary_cost_value + cost_opt_tolerance]
    linear_constraints += tv_constraints

    return cp.Problem(cp.Minimize(cp.sum(delta_var)), linear_constraints)


def solve_charge_action_for_step(step_idx, current_soc, full_price_profile, cfg, linearize=True):
    pred_horizon = min(horizon_len, len(full_price_profile) - step_idx)
    pred_price_profile = full_price_profile[step_idx:step_idx + pred_horizon]

    bess_charge_var = cp.Variable(pred_horizon)
    bess_soc_var = cp.Variable(pred_horizon + 1)

    mpc_constraints = [bess_soc_var[0] == current_soc]

    for target_state_index, target_soc in cfg['targets']:
        # Convert global target index to local horizon index at this step.
        if step_idx <= target_state_index <= step_idx + pred_horizon:
            local_idx = target_state_index - step_idx
            mpc_constraints += [bess_soc_var[local_idx] == target_soc]

    for t in range(pred_horizon):
        mpc_constraints += [
            bess_soc_var[t + 1] == bess_soc_var[t] + charge_coefficient * bess_charge_var[t],
            bess_charge_var[t] >= cfg['bess_charge_min'],
            bess_charge_var[t] <= cfg['bess_charge_max'],
        ]

    mpc_constraints += [
        bess_soc_var >= cfg['bess_soc_min'],
        bess_soc_var <= cfg['bess_soc_max'],
    ]

    primary_cost_expr = cp.sum(cp.multiply(pred_price_profile, bess_charge_var))
    problem = cp.Problem(cp.Minimize(primary_cost_expr), mpc_constraints)
    problem.solve()

    if not is_optimal_status(problem.status) or bess_charge_var.value is None:
        return 0.0

    if linearize and pred_horizon > 1:
        primary_value = problem.value
        if primary_value is None:
            return float(bess_charge_var.value[0])
        primary_cost_opt = float(np.asarray(primary_value).item())

        problem_with_linearity_bias = build_linear_problem(
            primary_cost_value=primary_cost_opt,
            action_var=bess_charge_var,
            pred_horizon=pred_horizon,
            base_constraints=mpc_constraints,
            cost_expr=primary_cost_expr,
        )
        problem_with_linearity_bias.solve()

        if is_optimal_status(problem_with_linearity_bias.status) and bess_charge_var.value is not None:
            return float(bess_charge_var.value[0])
    
    else:
        return float(bess_charge_var.value[0])

bess_soc = 0.3
bess_charge_action = 0.0
charge_coefficient = 0.1

net_cost = 0.0
cost_history = []
soc_history = [bess_soc]
action_history = []
cumulative_cost_history = []


for i in range(len(price_profile_1)):
    # 1. update bess soc and cost history with last action
    bess_charge_action = bess_charge_action if bess_charge_action is not None else 0.0
    bess_charge = charge_coefficient * bess_charge_action
    bess_soc += bess_charge
    bess_soc = np.clip(bess_soc, constraints['bess_soc_min'], constraints['bess_soc_max'])
    cost_history.append(bess_charge * price_profile_1[i])

    # 2. solve next action with lexicographic LP (cost first, smoothness second)
    bess_charge_action = solve_charge_action_for_step(
        step_idx=i,
        current_soc=bess_soc,
        full_price_profile=price_profile_1,
        cfg=constraints,
        linearize=linearize_actions,
    )

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
