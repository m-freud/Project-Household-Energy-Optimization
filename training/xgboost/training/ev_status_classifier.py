# EV status classifier for xgboost
# we want to train a classifier that yields the position of an EV for the next n timesteps
# n = remaining timesteps
# the mpc solver needs length 96 but we can pad the tail

# the classifier only predicts the next step, then starts again from there until the horizon is reached.
# paste this to enable src. imports
from pathlib import Path
import sys
import math

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from xgboost import XGBClassifier
from training.xgboost.features.ev_status_features import get_ev_status_features
from training.split import distinct_set_ignore_ev_status


distinct_ids = sorted(distinct_set_ignore_ev_status)
n_train = int(len(distinct_ids) * 0.8)
train_household_ids = distinct_ids[:n_train]
test_household_ids = distinct_ids[n_train:]

print(f"Total distinct household IDs: {len(distinct_ids)}")

print(f"Train household IDs: {train_household_ids}")
print(f"Test household IDs: {test_household_ids}")
print(f"Using curated distinct set size: {len(distinct_ids)}")

train = get_ev_status_features(train_household_ids)
test = get_ev_status_features(test_household_ids)

# Keep the matrix purely numeric; drop identity/string columns and target.
drop_columns = ["next_state", "household_id", "ev_key", "phase"]
X_train, y_train = train.drop(columns=drop_columns), train["next_state"]
X_test, y_test = test.drop(columns=drop_columns), test["next_state"]

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
model.save_model("ev_status_classifier.json")