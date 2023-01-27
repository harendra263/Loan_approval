import pandas as pd
from sklearn import preprocessing

class CategoricalFeatures:
    def __init__(self, dataframe, categorical_features, encoding_type, handle_na=False):
        """
        dataframe: pandas dataframe
        categorical_features: List of column names, e.g. ['ord1', 'ord2', ..]
        encoding_type: label, binary, ohe etc.
        handle_na: True/ False
        """
        self.dataframe = dataframe
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = {}

        if self.handle_na:
            for c in self.cat_feats:
                self.dataframe.loc[:, c] = self.dataframe.loc[:, c].astype(str).fillna("-999999")
        self.output_df = self.dataframe.copy(deep=True)

    
    def _label_encoding(self) -> pd.DataFrame:
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.output_df[c].values)
            self.output_df.loc[:, c] = lbl.fit_transform(self.dataframe[c].values)
            self.label_encoders[c] = lbl
        return self.output_df

    def fit_transform(self):
        if self.enc_type == 'label':
            return self._label_encoding()
        else:
            raise Exception("Encoding type not understood")
    
    def transform(self, df):
        if self.handle_na:
            for c in self.cat_feats:
                df.loc[:, c] = df.loc[:, c].astype(str).fillna('-999999')
        
        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items:
                df.loc[:, c] = lbl.transform(df[c].values)
            return df
