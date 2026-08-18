'''
Tune hypams for ridge with the new setup

we now have a clear and clean global train-test partition of 174/76

within the 176, we have target-specific train-test splits of about 60/40

we find optimal hypams like this:

pick a target e.g. base_load
and params for this model

train a model with these params on 60, cost function: rsme

test model on 40, cost function: total cost after a simulation,
where the other predictors are set to oracle.

'''
import sys
from pathlib import Path

repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))
from training.split.clean_split import PARTITIONS
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import root_mean_squared_error

train_ridge = PARTITIONS['ridge']['train']
test_ridge = PARTITIONS['ridge']['test']


def tune():
    pass


if __name__ == '__main__':
    tune()