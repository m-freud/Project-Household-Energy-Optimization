# create instance of xgb, random forest and ridge regression,
# classifier and regressor each, 
# print  model.__class__.__name__.lower() for each

from simulation.controllers.mpc.predictors.ml.model_interface import ModelLike
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeClassifier, Ridge

def print_model_names():
    models = [
        XGBClassifier(),
        XGBRegressor(),
        RandomForestClassifier(),
        RandomForestRegressor(),
        RidgeClassifier(),
        Ridge()
    ]

    for model in models:
        print(model.__class__.__name__.lower())

print_model_names()
