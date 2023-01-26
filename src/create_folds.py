import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":
    df = pd.read_parquet("parquet_files/loan.parquet")

    df['kfolds'] = -1

    df = df.sample(frac=1).reset_index(drop=True)
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.loan_status)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold
    
    df.to_parquet("input/train_folds.parquet", index=False)
