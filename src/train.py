import pandas as pd
import numpy as np
from sklearn import metrics
import os
import joblib
from . import dispatcher


TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")



FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    df = pd.read_parquet(TRAINING_DATA)

    df_train = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
    df_valid = df[df.kfold == FOLD].reset_index(drop=True)

    ytrain = df_train.enc_loan_status.values
    yvalid = df_valid.enc_loan_status.values

    df_train = df_train.drop(['enc_loan_status', 'kfold'], axis=1)
    df_valid = df_valid.drop(['enc_loan_status', 'kfold'], axis=1)

    # data is ready to train
    clf = dispatcher.MODELS[MODEL]
    clf.fit(df_train, ytrain)
    preds = clf.predict_proba(df_valid)[:, 1]
    print(metrics.roc_auc_score(yvalid, preds))

    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(df_train.columns, f"models/{MODEL}_{FOLD}_columns.pkl")




