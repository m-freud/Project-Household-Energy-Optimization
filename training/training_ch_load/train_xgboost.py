# paste this to enable src. imports
from pathlib import Path
import sys
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor, train
# paste this to enable src. imports

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))
from src.runtime_config import RuntimeConfig
from src.simulation.controllers.mpc.predictors.ml.model_config import ModelConfig
from training._archive.xgboost.feature_analysis import get_base_load_features, get_pv_gen_features, get_ev_status_features
from training.model_artifacts import write_training_params_manifest
from training._archive.xgboost.training.train_models import load_best_params_from_tuning, _feature_columns_for_target
from sklearn.metrics import root_mean_squared_error

from training.split.clean_split import PARTITIONS

base_load_test = PARTITIONS["inner"]["base_load"]["test"]
base_load_train_portugal = PARTITIONS["inner"]["base_load"]["train"]

repo_root = RuntimeConfig.ROOT_DIR

TUNING_RESULTS_CSV_PATH = Path(RuntimeConfig.ROOT_DIR / "training" / "xgboost" / "tuning" / "results.csv")
model_save_path_portugal = Path(__file__).parent / "models" / "xgboost_model_portugal176.json"
model_save_path_ch =  Path(__file__).parent / "models" / "xgboost_model_A.json"


params = load_best_params_from_tuning()
params = params["base_load"]

def train_ch():
    train_df_ch = pd.DataFrame()

    for parquet in (RuntimeConfig.ROOT_DIR / "training_ch_load" / "feature_dataframes").glob("*.parquet"):
        df = pd.read_parquet(parquet)
        train_df_ch = pd.concat([train_df_ch, df], ignore_index=True)


    model_A = XGBRegressor(
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
    )

    # rename 'load' column  to 'base_load' column
    train_df_ch.rename(columns={"load": "base_load"}, inplace=True)

    X_train = train_df_ch[_feature_columns_for_target("base_load")]


    y_train = train_df_ch["next_value"]

    model_A.fit(X_train, y_train)
    model_A.save_model(model_save_path_ch)

    print(f"Model A trained and saved to {model_save_path_ch}")


def train_portugal():
    train_df_portugal = get_base_load_features(base_load_train_portugal)
    model_A = XGBRegressor(
        learning_rate=params["learning_rate"],
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
    )

    X_train = train_df_portugal[_feature_columns_for_target("base_load")]
    y_train = train_df_portugal["next_value"]

    model_A.fit(X_train, y_train)
    model_A.save_model(model_save_path_portugal)

    print(f"Model A trained and saved to {model_save_path_portugal}")


def compare_models():
    # compare both models on the base load test set (PARTITIONS)
    feature_columns = _feature_columns_for_target("base_load")
    test_df = get_base_load_features(base_load_test)
    X_test = test_df[feature_columns]
    y_test = test_df["next_value"]

    model_ch = XGBRegressor()
    model_ch.load_model(model_save_path_ch)

    model_portugal = XGBRegressor()
    model_portugal.load_model(model_save_path_portugal)

    rmse_ch = root_mean_squared_error(y_test, model_ch.predict(X_test))
    rmse_portugal = root_mean_squared_error(y_test, model_portugal.predict(X_test))

    print(f"Model CH RMSE on base load test set:       {rmse_ch:.5f}")
    print(f"Model Portugal RMSE on base load test set: {rmse_portugal:.5f}")

if __name__ == "__main__":
    # train_ch()
    # train_portugal()
    compare_models()
