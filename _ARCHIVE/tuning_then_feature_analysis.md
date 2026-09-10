# Tuning Then Feature Analysis Workflow

This project follows a two-stage workflow per model family and per target:

1. Tune a small, reasonable hyperparameter grid first.
2. Use the best tuned settings to run feature relevance analysis.
3. Flag consistently weak features as drop candidates.

## Why this order

Feature relevance is sensitive to model capacity and regularization. If we analyze features with untuned defaults, we can overestimate or underestimate feature usefulness.

## Procedure

1. Build fold-wise tuning results for each model family.
2. Select best params per target using the lowest mean_score across folds.
3. Load those best params in training and feature-analysis scripts.
4. Generate per-fold and all-target feature relevance summaries.
5. Mark low-signal features as candidates for removal (not automatic deletion).

## Current implementation in this repo

- Random Forest tuning writes: training/random_forest/tuning/results.csv
- Ridge tuning writes: training/ridge_regression/tuning/results.csv
- Random Forest feature analysis reads tuned params from its tuning results.
- Ridge feature analysis reads tuned params from its tuning results.
- Training scripts for both RF and Ridge also read tuned params.

## Practical run order

1. Run tuning scripts.
2. Run feature analysis scripts.
3. Review summary CSVs and drop-candidate CSVs.
4. Update feature sets and rerun tuning + analysis if feature set changes.

## Domain note

For pv_gen, fold E is intentionally excluded in analysis/tuning where applicable due to non-PV household handling.
