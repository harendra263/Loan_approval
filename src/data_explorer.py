import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
        if c in df.select_dtypes(include=["float64", "int64"]).columns:
            df.loc[:, c].fillna(df[c].median(), inplace=True)
        elif c in df.select_dtypes(include='object').columns:
            df.loc[:, c].fillna(df[c].mode()[0], inplace=True)
    return df

def feature_selection_by_correlation(df: pd.DataFrame, threshold: int =0.8) ->pd.DataFrame:
    num_cols = df.select_dtypes(include= ['int64', 'float64']).columns
    corr = df[num_cols].corr()
    # Selecting upper triangle of the correlation matrix
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool_))
    # Find index of feature columns with correlation greater than a threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    # Drop features
    df= df.drop(df[to_drop], axis=1)
    return df


def correlation_plot(df: pd.DataFrame) -> None:
    print("Plot started")
    num_cols = df.select_dtypes(include= ['int64', 'float64']).columns
    corr = df[num_cols].corr()
    # Plot the correlation heatmap
    plt.figure(figsize=(15, 8))
    sns.heatmap(corr, annot=True, cmap='viridis')
    plt.savefig("input/Correlation.png")
    plt.show()
    plt.close()
    print("Correlation image saved.")



if __name__ == "__main__":
    df = pd.read_parquet("parquet_files/train.parquet")
    print(df.shape)

    missing_df = show_missing_pct(df=df).reset_index()
    col_to_drop = missing_df[missing_df.MissingPct >= 50]['level_0']

    df.drop(col_to_drop, axis=1, inplace=True)
    print(df.shape)

    print(get_nunique_value_from_categorical_col(df=df))
    cat_cols_to_drop = ["emp_title", "issue_d", "title", "zip_code", "addr_state", "earliest_cr_line", 
                        "last_pymnt_d", "last_credit_pull_d"]
    df.drop(cat_cols_to_drop, axis=1, inplace=True)
    print(df.shape)

    df = imputer(df=df)
    print(df.isnull().sum())

    # Plotting Correlation Plot and saving it to the img directory

    num_cols = df.select_dtypes(include=["float64", "int64"]).columns

    # Feature Selection by Correlation

    df = feature_selection_by_correlation(df=df)
    # correlation_plot(df=df)

    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].astype("category")
    df = lbl_encoding(df=df)

    var_df = variance_check(df=df).reset_index()
    print(var_df[var_df.Variance <= 0])
    df.drop(['policy_code', 'recoveries', 'collection_recovery_fee', 'enc_pymnt_plan',
            'enc_hardship_flag'], axis=1, inplace=True)

    print(df.shape)
    print(df.isnull().sum())

    df.to_parquet("input/train_data.parquet", index=False)

    



    