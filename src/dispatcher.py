from sklearn import ensemble
from sklearn import linear_model
import xgboost

MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "xgboost": xgboost.XGBClassifier(),
    "logistic_reg": linear_model.LogisticRegression(),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2)
    }