import pandas as pd
import joblib
import os



def predict(test_data_path, model_type, model_path):
    df = pd.read_parquet(test_data_path)
    predictions = None
    for FOLD in range(5):
        df = pd.read_parquet(test_data_path)
        cols = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}_columns.pkl"))
        clf = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}.pkl"))
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if FOLD==0:
            predictions = preds
        else:
            predictions += preds
        
    predictions /=5


if __name__ == "__main__":
    submission = predict(test_data_path="input/test_data.parquet",
                        model_path="models/",
                        model_type="randomforest")
    
