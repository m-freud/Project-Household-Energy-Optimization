import time
import numpy as np
import cvxpy as cp


# --------------------------
# Minimal HEMS-like settings
# --------------------------
T = 96                      # full day steps
H = 60                      # fixed MPC horizon
charge_coeff = 0.1
soc_min, soc_max = 0.0, 1.0
u_min, u_max = 0.0, 1.0
target_soc = 0.8
target_state_index = T      # global SoC index target

# toy price profile: expensive, cheap, very expensive
full_price = np.array([2.0] * 33 + [1.0] * 33 + [3.0] * 30, dtype=float)


def make_step_inputs(step_idx, full_price, horizon):
    # Build fixed-length parameter vectors so the compiled graph shape never changes.
    remaining = len(full_price) - step_idx
    active_len = min(horizon, remaining)

    price_vec = np.full(horizon, 1e3, dtype=float)      # high price for padded tail
    active_mask = np.zeros(horizon, dtype=float)        # 1 where horizon is active, else 0
    price_vec[:active_len] = full_price[step_idx:step_idx + active_len]
    active_mask[:active_len] = 1.0

    # Reachability guard vector for each predicted state index k in 0..H
    # rem_steps[k] = max steps left from global state (step_idx + k) to target_state_index
    rem_steps = np.maximum(
        0,
        target_state_index - (step_idx + np.arange(horizon + 1))
    ).astype(float)

    return price_vec, active_mask, rem_steps


def run_rebuild_every_step():
    soc = 0.3
    action = 0.0

    t0 = time.perf_counter()

    for step_idx in range(T):
        # apply previous action
        soc = np.clip(soc + charge_coeff * action, soc_min, soc_max)

        # rebuild full problem every step (slow pattern)
        price_vec, active_mask, rem_steps = make_step_inputs(step_idx, full_price, H)

        u = cp.Variable(H)
        s = cp.Variable(H + 1)

        constraints = [
            s[0] == soc,
            s[1:] == s[:-1] + charge_coeff * u,
            u >= u_min * active_mask,
            u <= u_max * active_mask,
            s >= soc_min,
            s <= soc_max,
            # Reachability guard (linear)
            s >= target_soc - rem_steps * charge_coeff * u_max,
        ]

        objective = cp.Minimize(price_vec @ u)
        prob = cp.Problem(objective, constraints)
        prob.solve(warm_start=True)

        action = float(u.value[0]) if (u.value is not None and prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)) else 0.0

    return time.perf_counter() - t0


def build_compiled_problem(horizon):
    # Variables
    u = cp.Variable(horizon)
    s = cp.Variable(horizon + 1)

    # Parameters (updated each step)
    soc0_p = cp.Parameter()
    price_p = cp.Parameter(horizon)
    active_p = cp.Parameter(horizon, nonneg=True)
    rem_steps_p = cp.Parameter(horizon + 1, nonneg=True)

    constraints = [
        s[0] == soc0_p,
        s[1:] == s[:-1] + charge_coeff * u,
        u >= u_min * active_p,
        u <= u_max * active_p,
        s >= soc_min,
        s <= soc_max,
        # Reachability guard (linear)
        s >= target_soc - rem_steps_p * charge_coeff * u_max,
    ]

    objective = cp.Minimize(price_p @ u)
    prob = cp.Problem(objective, constraints)

    return {
        "prob": prob,
        "u": u,
        "soc0_p": soc0_p,
        "price_p": price_p,
        "active_p": active_p,
        "rem_steps_p": rem_steps_p,
    }


def run_compiled_reuse():
    soc = 0.3
    action = 0.0

    ctrl = build_compiled_problem(H)
    prob = ctrl["prob"]

    t0 = time.perf_counter()

    for step_idx in range(T):
        # apply previous action
        soc = np.clip(soc + charge_coeff * action, soc_min, soc_max)

        # only update parameter values (fast pattern)
        price_vec, active_mask, rem_steps = make_step_inputs(step_idx, full_price, H)
        ctrl["soc0_p"].value = soc
        ctrl["price_p"].value = price_vec
        ctrl["active_p"].value = active_mask
        ctrl["rem_steps_p"].value = rem_steps

        prob.solve(warm_start=True)

        u_val = ctrl["u"].value
        action = float(u_val[0]) if (u_val is not None and prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)) else 0.0

    return time.perf_counter() - t0


if __name__ == "__main__":
    t_rebuild = run_rebuild_every_step()
    t_reuse = run_compiled_reuse()

    print(f"Rebuild every step: {t_rebuild:.4f} s")
    print(f"Compiled once/reused: {t_reuse:.4f} s")
    if t_reuse > 0:
        print(f"Speedup: {t_rebuild / t_reuse:.2f}x")

"""
Example run results (local):
Rebuild every step: 0.7366 s
Compiled once/reused: 0.2670 s
Speedup: 2.76x
"""
