
from pathlib import Path
import sys
# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))


from xgboost import XGBRegressor
from training.xgboost.features.ev_status_features import get_ev_status_features
from training.xgboost.features.pv_gen_features import get_pv_features
from training.split import distinct_set_ignore_ev_status


distinct_ids = sorted(distinct_set_ignore_ev_status)
n_train = int(len(distinct_ids) * 0.8)
train_household_ids = distinct_ids[:n_train]
test_household_ids = distinct_ids[n_train:]

train = get_pv_features(train_household_ids)
test = get_pv_features(test_household_ids)

print(f"Train rows: {len(train)} | Test rows: {len(test)}" )
print("samlple train rows:")
print(train.head(50))

