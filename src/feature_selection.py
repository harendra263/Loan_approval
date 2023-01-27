import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import train_test_split



class FeatureSelection:
    def __init__(self, df: pd.DataFrame, target_col: str) -> None:
        self.dataframe = df
        self.target_col = target_col
        self.cols = self.dataframe.columns
    

    def get_dataframe_details(self, feature_selection_method: str) ->pd.DataFrame:
        """
        feature_selection_method: takes values such 'Missing_Values' and 'Variance' as selection method
        and returns dataframe for the respective method
        """
        if feature_selection_method == "Missing_Values":
            missing_pct = pd.DataFrame(columns=["Missing_Pct"], index=[self.cols])
            for col in self.cols:
                missing_pct.loc[col, "Missing_Pct"] = (self.dataframe[col].isnull().sum())/ len(self.dataframe) * 100
            return missing_pct
        elif feature_selection_method == "Variance":
            variance_df = pd.DataFrame(columns=["Variance"], index=[self.cols])
            for i in self.cols:
                variance_df.loc[i, "Variance"] = (self.dataframe[i].var())
            return variance_df

        else:
            raise Exception("Selection method not understood")


    def apply_lasso(self) -> list[str]:
        X = self.dataframe.drop(self.target_col, axis=1)
        y = self.dataframe[self.target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        sel = SelectFromModel(LogisticRegression(C=0.5, penalty='l1', solver='liblinear', random_state=10))
        sel.fit(X_train, y_train)
        return X_train.columns[(sel.estimator_.coef_ == 0).ravel().tolist()]

    def get_correlation_matrix(self, cols: list[str]) -> None:
        pass