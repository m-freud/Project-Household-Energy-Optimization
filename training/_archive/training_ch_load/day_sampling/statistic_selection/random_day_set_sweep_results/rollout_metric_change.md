# Rollout Metric Change

The rollout evaluation was corrected between these two sweeps.

## Old sweep (ncand 201)

`horizon_bucket_h` measured the RMSE over the **entire forecast prefix from horizon 1 through h**:

RMSE(1...h)

For example, `horizon_bucket_18` contained the cumulative trajectory error over forecast steps 1–18.

These results remain valid as **cumulative rollout / trajectory-error metrics**, but must not be interpreted as horizon-specific forecast errors.

## Patched sweep

`horizon_bucket_h` now measures the RMSE of predictions made **exactly h steps ahead**:

RMSE_h = RMSE(y_hat[t+h|t], y[t+h])

For example, `horizon_bucket_18` now measures only the error of forecasts 18 timesteps (4.5 hours) ahead.

Squared errors are collected across forecast origins and evaluated as RMSE after pooling, rather than averaging already-computed RMSE values.

## Consequence

Results from the two sweeps are **not directly comparable for the horizon-bucket metrics**.

- Old sweep: cumulative forecast quality up to horizon h
- New sweep: forecast quality specifically at horizon h

The standard one-step `rmse`, `rmse_20_loads`, and downstream `net_cost` metrics are unaffected by this change.