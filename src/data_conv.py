import pandas as pd

"""Dividing the raw data into two parts .i.e. Train and Test"""

if __name__ == "__main__":
    df = pd.read_csv("input/loan.csv", low_memory=False)

    df_train = df.iloc[:565167, :]
    df_test = df.iloc[565167:1000000, :]

    df_train.to_parquet("parquet_files/train.parquet", index=False)
    df_test.to_parquet("parquet_files/test.parquet", index=False)
    # df.to_parquet(path="parquet_files/loan.parquet")