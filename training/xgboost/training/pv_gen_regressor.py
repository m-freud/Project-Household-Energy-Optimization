
from pathlib import Path
import sys
import math
# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))


from xgboost import XGBRegressor  # noqa: E402
from training.xgboost.features.pv_gen_features import get_pv_features  # noqa: E402
from src.config import Config  # noqa: E402
import numpy as np  # noqa: E402

train_household_ids = list(Config.H_SET_TRAINING)
test_household_ids = list(Config.H_SET_TESTING)

train = get_pv_features(train_household_ids)
test = get_pv_features(test_household_ids)

drop_columns = ["household_id", "next_value"]

X_train, y_train = train.drop(columns=drop_columns), train["next_value"]
X_test, y_test = test.drop(columns=drop_columns), test["next_value"]


model = XGBRegressor()

model.fit(X_train, y_train)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	y_true = np.asarray(y_true, dtype=float)
	y_pred = np.asarray(y_pred, dtype=float)

	ss_res = float(np.sum((y_true - y_pred) ** 2))
	ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

	if ss_tot == 0.0:
		return 0.0
	return 1.0 - (ss_res / ss_tot)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)))


y_pred_test = model.predict(X_test)

r2 = _r2_score(y_test.to_numpy(), y_pred_test)
mae = _mae(y_test.to_numpy(), y_pred_test)
rmse = _rmse(y_test.to_numpy(), y_pred_test)

pv_window = getattr(Config, "PV_GENERATION_WINDOW_ALLOWED", None)
if pv_window is None:
	pv_window = getattr(Config, "PV_GENERATION_WINDOW_OBSERVED", None)

if pv_window is None:
	raise ValueError("Config must define PV_GENERATION_WINDOW_ALLOWED or PV_GENERATION_WINDOW_OBSERVED")

daylight_start = int(pv_window["earliest_start"])
daylight_end = int(pv_window["latest_end"])

daylight_mask = (test["timestep"] >= daylight_start) & (test["timestep"] <= daylight_end)
daylight_count = int(daylight_mask.sum())

if daylight_count > 0:
	y_test_day = y_test.loc[daylight_mask].to_numpy()
	y_pred_day = y_pred_test[daylight_mask.to_numpy()]
	r2_day = _r2_score(y_test_day, y_pred_day)
	mae_day = _mae(y_test_day, y_pred_day)
	rmse_day = _rmse(y_test_day, y_pred_day)
else:
	r2_day = math.nan
	mae_day = math.nan
	rmse_day = math.nan


print(f"Test R2: {r2:.6f}")
print(f"Test MAE: {mae:.6f}")
print(f"Test RMSE: {rmse:.6f}")
print(f"Daylight window: [{daylight_start}, {daylight_end}] | Test daylight rows: {daylight_count}")
print(f"Daylight Test R2: {r2_day:.6f}")
print(f"Daylight Test MAE: {mae_day:.6f}")
print(f"Daylight Test RMSE: {rmse_day:.6f}")
print(f"Train rows: {len(train)} | Test rows: {len(test)}")
print(f"Train households: {train_household_ids}")
print(f"Test households: {test_household_ids}")

# save model
root = Config.ROOT_DIR
model_path = Config.XGB_PV_GEN_MODEL_PATH
model.save_model(model_path)
