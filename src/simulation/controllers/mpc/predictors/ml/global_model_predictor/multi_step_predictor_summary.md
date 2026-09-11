# Forecast Predictor Architecture – Summary

## Current setup

The existing predictor generates the future load profile **recursively**: predict the next load value, feed that prediction back into lag/history features, and repeat over the forecast horizon.

A clearer class name is:

```python
RecursiveMLPredictor
```

The relevant weakness is that recursive forecasting can propagate errors through the trajectory and create horizon-dependent accuracy trade-offs.

## New approach: CompositeMLPredictor

The planned alternative uses **separate direct models for selected forecast horizons**, for example:

```python
horizons = [4, 8, 16, 32, 48, 64, 96]
```

Each model directly learns `f_h(X_t) -> y_(t+h)`. The anchor predictions are then combined into a complete forecast profile, initially using **linear interpolation**.

Suggested class name:

```python
CompositeMLPredictor
```

This gives a clean architectural comparison:

```text
RecursiveMLPredictor
        vs.
CompositeMLPredictor
```

In the README, the second method can be described more formally as **direct multi-horizon forecasting using separate anchor-horizon models and interpolation**.

Advantages:
- avoids recursive error propagation between anchor predictions;
- allows each model to specialize in one forecast horizon;
- requires only a small number of direct models;
- keeps individual horizon predictions interpretable;
- linear interpolation keeps the experiment simple;
- directly tests whether recursive forecasting contributes to the observed horizon-dependent error trade-offs.

The primary evaluation criterion remains **downstream MPC net cost**. Forecast RMSE metrics are diagnostics.

## Alternative: one shared direct multi-horizon model

A third architecture is a **single model shared across forecast horizons**.

Instead of separate models such as `f_4(X_t)`, `f_8(X_t)`, and `f_16(X_t)`, train:

`f(X_t, h) -> y_(t+h)`

where forecast horizon `h` and/or target timestep is included as an input feature.

Possible inputs include current load features, lags, history statistics, current timestep, forecast horizon, target timestep, and target-time sin/cos features.

### Potential advantages

**Shared learning across horizons.** Nearby horizons are related prediction problems. A shared model can learn common structure rather than forcing every horizon-specific model to rediscover it.

**More effective training information per model.** One model sees observations from many horizons, which may help when independent training data is limited.

**Arbitrary forecast horizons.** Horizon becomes an input variable, so one model can in principle predict many future steps without a separately trained model for each one.

**Potentially more consistent forecasts.** Independent horizon models can disagree or produce discontinuities. A shared model may learn a more coherent relationship between horizon and predicted load.

**Simpler model management.** Only one trained model needs to be stored, loaded, versioned, and deployed.

**Generalization across horizons.** Patterns useful at several forecast distances can be represented once and reused.

### Disadvantages

- Training data can become much larger because each state can generate samples for many future horizons.
- Producing a complete forecast may still require many model evaluations.
- One model has to represent several forecast-distance regimes simultaneously.
- Horizon-specific behavior is less explicit and less interpretable.
- It becomes harder to isolate whether an improvement comes specifically from removing recursion or from shared learning.
- For the current research question, separate anchor models provide a cleaner experiment.

The shared model is therefore a good **future-work/backlog** item rather than a requirement for the current project.

It could also be trained only on the same anchor horizons as the composite predictor, allowing a controlled comparison:

```text
Recursive forecasting
        ↓
Separate direct horizon models
        ↓
Shared direct horizon model
```

## Recommended current scope

Implement and evaluate:

```text
RecursiveMLPredictor
vs.
CompositeMLPredictor
```

Keep the shared direct multi-horizon model as an explicitly considered alternative.

Possible README wording:

> An alternative architecture would use a single shared direct multi-horizon model, with forecast horizon or target timestep provided as an additional feature. Such a model could exploit common structure across forecast horizons and reduce the number of separately maintained models. This approach was left for future work in favor of separate anchor-horizon models, which provide a simpler and more interpretable test of recursive error propagation.

The key experimental question remains:

> **Does replacing recursive rollout forecasting with direct anchor-horizon predictions improve downstream MPC performance?**
