import pandas as pd
import numpy as np

def show_missing_pct(df: pd.DataFrame) ->pd.DataFrame:
    col = df.columns.to_list()
    missing_pct = pd.DataFrame(columns=["MissingPct"], index=[col])
    for i in col:
        missing_pct.loc[i, "MissingPct"] = (df[i].isnull().sum()/ len(df)) * 100
    return missing_pct


def variance_check(df: pd.DataFrame) ->pd.DataFrame:
    col = df.columns.to_list()
    variance_df = pd.DataFrame(columns=["Variance"], index=[col])
    for i in col:
        variance_df.loc[i, "Variance"] = (df[i].var())
    return variance_df

def get_nunique_value_from_categorical_col(df: pd.DataFrame) ->pd.DataFrame:
    cat_cols = df.select_dtypes(include="object").columns
    nunique_cat_vals = pd.DataFrame(columns=["Nunique"], index=[cat_cols])
    for col in cat_cols:
        nunique_cat_vals.loc[col, "Nunique"] = df[col].nunique()
    return nunique_cat_vals


def lbl_encoding(df: pd.DataFrame) ->pd.DataFrame:
    cat_cols = df.select_dtypes(include="category").columns
    for c in cat_cols:
        df[f'enc_{c}'] = df[c].cat.codes
        df.drop(c, axis=1, inplace=True)
    return df

def imputer(df: pd.DataFrame) ->pd.DataFrame:
    for c in df.columns:
        if c in df.select_dtypes(include="int").columns:
            df.loc[:, c].fillna(df[c].median(), inplace=True)
        elif c in df.select_dtypes(include='object').columns:
            df.loc[:, c].fillna(df[c].mode()[0], inplace=True)
        return df


if __name__ == "__main__":
    df = pd.read_parquet("parquet_files/train.parquet")
    print(df.shape)

    missing_df = show_missing_pct(df=df).reset_index()
    col_to_drop = missing_df[missing_df.MissingPct >= 50]['level_0']

    df.drop(col_to_drop, axis=1, inplace=True)
    print(df.shape)

    print(get_nunique_value_from_categorical_col(df=df))
    cat_cols_to_drop = ["emp_title", "issue_d", "title", "zip_code", "addr_state", "earliest_cr_line", 
                        "last_pymnt_d", "next_pymnt_d", "last_credit_pull_d"]
    df.drop(cat_cols_to_drop, axis=1, inplace=True)
    print(df.shape)

    df = imputer(df=df)
    print(df.isnull().sum())
    
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].astype("category")
    df = lbl_encoding(df=df)
    print(df.head())


    



    