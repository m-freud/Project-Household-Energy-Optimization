# EV status classifier for xgboost
# we want to train a classifier that yields the position of an EV for the next n timesteps
# n = remaining timesteps
# the mpc solver needs length 96 but we can pad the tail

# the classifier only predicts the next step, then starts again from there until the horizon is reached.
# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from src.config import Config  # noqa: E402

from xgboost import XGBClassifier  # noqa: E402
from training._features.ev_status_features import get_ev_status_features  # noqa: E402


train_household_ids = list(Config.H_SET_TRAINING)
test_household_ids = list(Config.H_SET_TESTING)

train = get_ev_status_features(train_household_ids)
test = get_ev_status_features(test_household_ids)

feature_columns = Config.XGB_FEATURES["EV_STATUS"]

for f in feature_columns:
	if f not in train.columns or f not in test.columns:
		raise ValueError(f"Missing feature '{f}' in features df.")

X_train, y_train = train[feature_columns], train["next_state"]
X_test, y_test = test[feature_columns], test["next_state"]

model = XGBClassifier(
	n_estimators=300,
	max_depth=6,
	learning_rate=0.05,
	subsample=0.9,
	colsample_bytree=0.9,
	objective="multi:softprob",
	num_class=3,
	random_state=42,
	eval_metric="mlogloss",
)
model.fit(X_train, y_train)

# test model
accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {accuracy}")
print(f"Train rows: {len(train)} | Test rows: {len(test)}")

# save model
model_path = Config.XGB_EV_STATUS_MODEL_PATH
model.save_model(model_path)