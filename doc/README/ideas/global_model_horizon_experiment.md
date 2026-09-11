# Global Model Horizon Experiment

Train one global model per target instead of one model per horizon.

The target horizon is provided as an additional feature:

```python
prediction_horizon = 1, 2, 4, 8, 16, 32, 64
```

At runtime, use the same model for all selected horizons and interpolate the resulting anchor predictions.

Compare:

- Dense horizon set versus sparse horizon set
- Linear interpolation
- Recursive interpolation between selected horizon anchors
- Direct prediction quality for short and long horizons
- Full-profile RMSE/MAE instead of only single-step RMSE
- Runtime and number of model calls

Example horizon sets:

```python
[1, 2, 4, 8, 16, 32, 48, 64]
[1, 2, 4, 8, 16, 24, 32, 48, 64]
[1, 2, 4, 8, 16, 32, 64, 80, 96]
```

Hypothesis:

A global horizon-conditioned model may reduce model storage and training complexity while retaining most of the accuracy of separate horizon models. Very long horizons such as 80 or 96 may be unreliable because they have fewer useful training examples and substantially higher uncertainty.

Recursive interpolation remains useful when only a sparse set of horizons is trained. It should be compared against linear interpolation, especially for regression targets. For classification targets, linear interpolation is disabled because it can produce invalid intermediate classes.
