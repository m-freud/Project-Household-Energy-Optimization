# Moving Average Calculation

This document describes the exact equations implemented in [src/simulation/controllers/mpc/predictors/ma3/moving_average.py](src/simulation/controllers/mpc/predictors/ma3/moving_average.py).

## Markdown Math Support

Yes, Markdown supports math in common tooling used here:

- VS Code Markdown preview (with math-enabled rendering)
- GitHub-flavored Markdown environments with LaTeX math enabled

Math is written with inline `$...$` and display `$$...$$` blocks.

## Terminology

This document uses the following distinction consistently:

- horizon: how far the forecast looks ahead
- range: how far a forward-applied effect remains active
- window: how far the predictor looks back into history

Applied here:

- forecast horizon: total number of future steps predicted
- trend window: backward lookback used to estimate recent trend
- trend range: number of future steps over which the trend effect is applied
- persistence range: number of future steps over which persistence remains active, or for exponential mode, the approximate handoff range

## Symbols

- $k \in \{1,\dots,H\}$: forecast step index
- $H$: forecast horizon
- $x_t$: latest observed value in history
- $y_k$: forecasted value at step $k$
- $L_s$: short window size (`short_window`)
- $L_\ell$: long window size (`long_window`)
- $w_s$: short-window weight (`short_weight`)
- $w_\ell = 1 - w_s$: long-window weight
- $\beta \in [0,1]$: trend blend (`trend_weight` after clamping)
- $R$: trend range (`trend_range`)
- $h$: persistence range (`persistence_horizon` in code)
- $c \in [0,1]$: constant persistence alpha (`persistence_constant_alpha`)

## 1. Seeded Recursive Series

A working series is seeded from recent history and then extended with predictions.

If history has at least $L_\ell$ values, take the last $L_\ell$ values.
Otherwise, left-pad with the oldest available value (or default when empty) until length is $L_\ell$.

## 2. Recent Trend Estimate

Using the tail of length $n = \min(\text{len(history)}, \max(2,\text{trend\_window}))$, define

$$
m = \frac{1}{n-1}\sum_{i=2}^{n}(z_i - z_{i-1}),
$$

where $z_1,\dots,z_n$ are the tail values.

So $m$ is the mean first-difference over the recent window.

## 3. Short/Long Moving-Average Prediction at Step $k$

From the recursively updated series at step $k$:

$$
\text{MA}^{(s)}_k = \frac{1}{L_s}\sum_{j=1}^{L_s} s_{k,j},
$$

$$
\text{MA}^{(\ell)}_k = \frac{1}{L_\ell}\sum_{j=1}^{L_\ell} \ell_{k,j},
$$

and

$$
\hat{y}^{\text{MA}}_k = w_s\,\text{MA}^{(s)}_k + (1-w_s)\,\text{MA}^{(\ell)}_k.
$$

## 4. Trend-Persistence Target

Let

$$
r_k = \min(\max(0, k-1), R).
$$

Current-value anchor:

$$
v_0 =
\begin{cases}
x_t, & \text{if history exists} \\
\text{default}, & \text{otherwise}
\end{cases}
$$

Trend target:

$$
\hat{y}^{\text{trend}}_k = v_0 + r_k\,m.
$$

Blend current-value persistence with trend target:

$$
\hat{y}^{\text{persist}}_k = (1-\beta)\,v_0 + \beta\,\hat{y}^{\text{trend}}_k
= v_0 + \beta\,r_k\,m.
$$

## 5. Persistence Fade Coefficient $\alpha_k$

### Mode: `none`

$$
\alpha_k = 1.
$$

### Mode: `constant`

$$
\alpha_k =
\begin{cases}
c, & k \le h \\
1, & k > h
\end{cases}
$$

### Mode: `linear`

$$
\alpha_k =
\begin{cases}
0, & k = 1 \\
1, & h = 1 \\
\min\left(1,\frac{k-1}{h-1}\right), & \text{otherwise}
\end{cases}
$$

### Mode: `exponential`

$$
\tau = \frac{h-1}{\ln 10},
$$

$$
\alpha_k =
\begin{cases}
0, & k = 1 \\
1, & h = 1 \\
\min\left(1, 1 - e^{-(k-1)/\tau}\right), & \text{otherwise}
\end{cases}
$$

Here $\tau$ is the exponential time constant. In this implementation it is not tuned independently; it is derived from $h$ so that the MA weight reaches about $90\%$ at step $h$:

$$
1 - e^{-(h-1)/\tau} = 1 - e^{-\ln 10} = 0.9.
$$

So `persistence_horizon` in code, interpreted as persistence range in the UI/documentation, means "roughly the 90% handoff point" rather than a hard cutoff.

## 6. Final Forecast Equation

At each step $k$:

$$
y_k = (1-\alpha_k)\,\hat{y}^{\text{persist}}_k + \alpha_k\,\hat{y}^{\text{MA}}_k.
$$

Equivalent compact form:

$$
y_k = (1-\alpha_k)\left(v_0 + \beta r_k m\right) + \alpha_k\,\hat{y}^{\text{MA}}_k.
$$

After computing $y_k$, it is appended to the series and used recursively for subsequent moving-average calculations.
