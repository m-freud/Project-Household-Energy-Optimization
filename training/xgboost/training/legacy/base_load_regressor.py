
from pathlib import Path
import sys
# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))


from xgboost import XGBRegressor  # noqa: E402
from training._features.base_load_features import get_base_load_features  # noqa: E402
from runtime_config import RuntimeConfig  # noqa: E402
import numpy as np  # noqa: E402

train_household_ids = list(RuntimeConfig.H_SET_TRAINING)
test_household_ids = list(RuntimeConfig.H_SET_TESTING)

train = get_base_load_features(train_household_ids)
test = get_base_load_features(test_household_ids)

feature_columns = RuntimeConfig.XGB_FEATURES["BASE_LOAD"]

for f in feature_columns:
	if f not in train.columns or f not in test.columns:
		raise ValueError(f"Missing feature '{f}' in features df.")

X_train, y_train = train[feature_columns], train["next_value"]
X_test, y_test = test[feature_columns], test["next_value"]

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

print(f"R2: {r2}, MAE: {mae}, RMSE: {rmse}")    
print(f"Train rows: {len(train)} | Test rows: {len(test)}")
print(f"Train households: {train_household_ids}")
print(f"Test households: {test_household_ids}")

# save model
root = RuntimeConfig.ROOT_DIR
model_path = RuntimeConfig.XGB_BASE_LOAD_MODEL_PATH
model.save_model(model_path)