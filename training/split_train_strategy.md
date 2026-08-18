# Leakage-Safe Model Tuning & Evaluation

## 1. Global Train/Test Split

Households can share equivalent **base-load or PV profiles**, creating potential leakage.

Build a graph:

- Node = household
- Edge = equivalent base-load **or** PV profile
- Connected components must remain together

This produces:

- **Global train:** 174 households
- **Global test:** 76 households
- PV prevalence: ~80% in both sets

The global test set remains completely untouched until final evaluation.

---

## 2. Target-Specific Validation Splits

Models are trained separately for:

- Base load
- PV generation
- EV1 status
- EV2 status

Within the 174-household global training set, create a new train/validation split for each target.

Only equivalence relevant to that target needs to be respected.

Example for base load:

- Equivalent base-load profiles remain together
- Split profile groups roughly **3:2**
- EV equivalence does not matter here (->random overlap assumption / infeasibility)

This gives:

**inner train → inner validation → global test**

---

## 3. Hyperparameter Tuning

For every hyperparameter combination:

1. Train the candidate model on the **inner training set**
2. Insert it into the MPC predictor
3. Set all other predictions to **oracle values**
4. Run the predefined control scenarios on the **inner validation households**
5. Calculate average end-of-day net cost

Therefore:

> Hyperparameters are selected based on **downstream control performance**, not prediction RMSE.

Using oracle predictions for the other targets isolates the control impact of the predictor currently being tuned.

---

## 4. Select Best Hyperparameters

For each prediction target:

> Best hyperparameters = configuration producing the lowest average validation EOD cost.

This is done independently for:

- Base load
- PV
- EV1
- EV2

---

## 5. Retrain Final Models

After hyperparameter selection, retrain each winning model using **all 174 global training households**.

The 76 global test households are still untouched.

---

## 6. Final Bake-Off

Combine the final trained predictors into the complete MPC:

**Load + PV + EV1 + EV2 → MPC → control actions**

Run the predefined scenarios across the **76 global test households**.

Primary KPI:

> **Average end-of-day net cost / cost saving**

This directly answers the actual research question:

> **Which forecasting approach produces the best downstream control performance?**