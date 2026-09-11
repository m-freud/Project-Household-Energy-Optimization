# XGB MPC — To-Do

## 1  Performance: make XGBPredictor fast enough to use

The recursive rollout (96 steps × one `pd.DataFrame` + `model.predict` call each) is the main bottleneck.

- [ ] If per-step recursion is unavoidable (base-load depends on previous predictions), batch what can be batched (time features, lag arrays) and only iterate over the truly sequential part
- [x] Add inference-time shortcuts before the model call where the answer is effectively deterministic:
  - PV: outside the daylight window, skip `model.predict` and force `pv_gen = 0`
  - EV status: outside commute windows / stable phases, persist the last known state instead of re-predicting
  - Base load: look for similarly stable periods or fallback rules that can skip the regressor on obvious steps
- [x] Stop predicting beyond the useful horizon when the tail is synthetic; pad defaults after timestep 96 instead of extending expensive recursive rollout
- [ ] Consider a hybrid horizon strategy: use XGB only for the near-term steps and switch to a cheaper history-average or cached baseline further out, where long-horizon XGB is highly speculative anyway
- [ ] Reduce per-step object churn in the recursive path by precomputing reusable feature rows / arrays and avoiding repeated dict-to-list projection where possible
- [ ] Move feature-schema validation out of the hot loop: validate required columns once, then reuse the fixed feature order without re-checking every timestep
- [ ] Benchmark before/after; target sub-second per household per horizon

---

## 2  Full review: training scripts and runtime predictors

Make sure training and inference are exactly aligned and the code is clean.

- [ ] Verify `MODEL_FEATURE_COLUMNS` in every runtime predictor matches the training DataFrame column order exactly (already tested by parity script — keep that test green)
- [ ] Confirm the EV-status recursive prediction index alignment (off-by-one risk between `prediction_index` and the returned at-home sequences)
- [ ] Review `_regression.py` helper names vs. what `base_load_features.py` and `pv_gen_features.py` actually import — resolve any alias confusion
- [ ] Remove dead code in training scripts (unused imports, old `root`-based path constructions)
- [ ] Ensure all three `train_all.py` paths use `Config.H_SET_TRAINING` / `Config.H_SET_TESTING` consistently

---

## 3  Model tuning

Train better models; don't change the feature set until tuning is done.

### All three models
- [ ] Grid-search `max_depth` (4, 6, 8), `learning_rate` (0.01, 0.05, 0.1), `n_estimators` (200, 500, 1000) with early stopping on a held-out validation fold
- [ ] Use `Config.H_SET_TESTING` strictly as the final holdout — tune on `Config.H_SET_TRAINING` only

### PV generation regressor
- [ ] Apply a daylight-window mask to predictions at inference time: force `next_pv_gen = 0` for any timestep outside `Config.PV_GENERATION_WINDOW_ALLOWED`
- [ ] Evaluate whether a separate model for the daylight window vs. outside improves accuracy

### Base-load regressor
- [ ] Consider a rolling-window cross-validation over the 96-step day profile rather than a random train/test split

### EV-status classifier
- [ ] Evaluate per-class accuracy (home / commuting / station) — the classifier may be systematically wrong on the `station` class
- [ ] Try `subsample` and `colsample_bytree` sweeps to reduce overfitting on the small household set

---

## 4  Pre-presentation state & README

Get to a clean, demonstrable checkpoint before the presentation.

- [ ] Run the full test-set benchmark for all scenarios (`default_scenario`, `stressed_low_start`, remaining) with both `mpc_xgb` and `mpc_hybrid` so the dashboard shows a complete side-by-side
- [ ] Write a concise **README summary** covering:
  - Project goal (household energy cost minimisation with MPC)
  - Data: 250 households, 96 timesteps/day, EV + BESS + PV + base load
  - Architecture: simulation engine → MPC controller → XGBoost predictors (base load, PV, EV status)
  - Key result: XGB-MPC beats hybrid moving-average MPC on 6/8 test households, v1 models, no tuning
  - How to run: `train_all.py`, `simulation.py --controllers mpc_xgb --households test_set`, dashboard
- [ ] Draw a one-page **system diagram** showing:
  - Data layer (SQLite) → Simulation → MPC Controller → Predictor (XGBPredictor) → three sub-models
  - Training pipeline (feature builders → XGBoost → saved `.json` models → runtime loaders)
  - Feedback loop: EV-status prediction feeds base-load prediction feeds MPC optimisation
