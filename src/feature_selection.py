import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant



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

    def calculate_information_gain(self) ->pd.DataFrame:
        X = self.dataframe.drop(self.target_col, axis=1)
        y = self.dataframe[self.target_col]

        # calculate information gain
        info_gain = mutual_info_classif(X, y)

        # Add information gain to dataframe
        info_gain_df = pd.DataFrame({'feature': X.columns, "info_gain": info_gain})
        # Sort the info gain dataframe
        info_gain_df = info_gain_df.sort_values("info_gain", ascending=False)
        return info_gain_df

    def feature_selection_by_correlation(self, threshold: int =0.8) ->pd.DataFrame:
        num_cols = self.dataframe.select_dtypes(include= ['int64', 'float64']).columns
        corr = self.dataframe[num_cols].corr()
        # Selecting upper triangle of the correlation matrix
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool_))
        # Find index of feature columns with correlation greater than a threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        # Drop features
        self.dataframe= self.dataframe.drop(self.dataframe[to_drop], axis=1)
        return self.dataframe
    
    def correlation_plot(self, df: pd.DataFrame) -> None:
        print("Plot started")
        num_cols = self.dataframe.select_dtypes(include= ['int64', 'float64']).columns
        corr = self.dataframe[num_cols].corr()
        # Plot the correlation heatmap
        plt.figure(figsize=(15, 8))
        sns.heatmap(corr, annot=True, cmap='viridis')
        plt.savefig("input/Correlation.png")
        plt.show()
        plt.close()
        print("Correlation image saved.")

    def forward_selection(self, significance_level=0.05) ->list[str]:
        initial_features = self.dataframe.columns.to_list()
        best_features = []
        while (len(initial_features) > 0):
            remaining_features = list(set(initial_features) - set(best_features))
            new_pval = pd.Series(index=remaining_features)
            for new_column in remaining_features:
                model = OLS(self.target_col, add_constant(self.dataframe[best_features + [new_column]])).fit()
                new_pval[new_column]= model.pvalues[new_column]
            min_p_value = new_pval.min()
            if min_p_value < significance_level:
                best_features.append(new_pval.idxmin())
            else:
                break
        return best_features
