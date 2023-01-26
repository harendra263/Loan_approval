import pandas as pd

"""Dividing the raw data into two parts .i.e. Train and Test"""

if __name__ == "__main__":
    df = pd.read_parquet("parquet_files/loan.parquet")

    df = df[df.loan_status.isin(["Default", "Fully Paid"])]

    df_train = df.iloc[:665167, :]
    df_test = df.iloc[665167:,:]
    print(f"train has {df_train.shape[0]} rows and {df_train.shape[1]} columns")
    print(f"test has {df_test.shape[0]} rows and {df_test.shape[1]} columns")

    df_train.to_parquet("parquet_files/train.parquet", index=False)
    df_test.to_parquet("input/test_data.parquet", index=False)
    # df.to_parquet(path="parquet_files/loan.parquet")