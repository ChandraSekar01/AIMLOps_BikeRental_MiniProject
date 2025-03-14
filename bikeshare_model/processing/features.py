from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X_trans = X.copy()

        # Find the number of NaN entries in the `weekday` column, and get their row indices
        wkday_null_idx = X[X['weekday'].isnull() == True].index   # .isna()
        # print("Number of missing values: {}".format(wkday_null_idx))
        # Use the `dteday` column to extract day names
        X['dteday'] = pd.to_datetime(X['dteday'])
        # X_transformed['day_name'] = X_transformed['dteday'].dt.day_name()
        X.loc[wkday_null_idx, 'weekday'] = X.loc[wkday_null_idx, 'dteday'].dt.day_name().apply(lambda x: x[:3])

        return X



class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self):
        self.most_frequent_value = None

    def fit(self, X, y=None):
        self.most_frequent_value = X['weathersit'].mode()[0]
        # print(f"Most frequent category : {self.most_frequent_value}")
        return self

    def transform(self, X):
        # X['weathersit'].fillna(self.most_frequent_value, inplace=True) # 'Clear
        X['weathersit'] = X['weathersit'].fillna(self.most_frequent_value)

        return X



# class Mapper(BaseEstimator, TransformerMixin):
#     """
#     Ordinal categorical variable mapper:
#     Treat column as Ordinal categorical variable, and assign values accordingly
#     """

#     def __init__(self, column_mappings):
#         self.column_mappings = column_mappings

#     def fit(self):
#         return self

#     def transform(self, X):
#         for column, mapping in self.column_mappings.items():
#             if column in X.columns:
#                 X['hr'] = X['hr'].apply(lambda x: mapping[x])

#         return X

class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper."""

    def __init__(self, variable: str, mapping: dict):

        if not isinstance(variable, str):
            raise ValueError("variables should be a str")

        self.variable = variable
        self.mapping = mapping

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_map = X.copy()
        # print(self.variables)
        # print(self.mappings)
        X_map[self.variable] = X_map[self.variable].map(self.mapping).astype('int')

        return X_map



class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self, columns):
        self.columns = columns
        self.lower_bounds = {}
        self.upper_bounds = {}

    def fit(self, X, y=None):
        for column in self.columns:
            q1 = X.describe()[column].loc['25%']
            q3 = X.describe()[column].loc['75%']
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            self.lower_bounds[column] = lower_bound
            self.upper_bounds[column] = upper_bound
            # self.lower_bound = lower_bound
            # self.upper_bound = upper_bound
        return self

    def transform(self, X):
        # Handle outliers for each specified column
        for column in self.columns:
            lower_bound = self.lower_bounds[column]
            upper_bound = self.upper_bounds[column]
            # Apply the bounds
            X[column] = np.clip(X[column], lower_bound, upper_bound)
        # colm = self.columns
        # for i in X.index:
        #     if X.loc[i,colm] > self.upper_bound:
        #         X.loc[i,colm]= self.upper_bound
        #     if X.loc[i,colm] < self.lower_bound:
        #         X.loc[i,colm]= self.lower_bound
        
        return X



class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False)

    def fit(self, X, y=None):
        self.encoder.fit(X[['weekday']])
        return self

    def transform(self, X):
        encoded_weekday = self.encoder.transform(X[['weekday']])
        enc_wkday_features = self.encoder.get_feature_names_out(['weekday'])

        # encoded_df = pd.DataFrame(encoded_weekday, columns=enc_wkday_features, index=X.index)

        X_transformed = X.copy()
        # X_transformed = pd.concat([X_transformed, encoded_df], axis=1)

        X_transformed[enc_wkday_features] = encoded_weekday

        return X_transformed


# Custom transformer to drop specific columns
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        # Nothing to fit, just return self
        return self

    def transform(self, X):
        # Drop the specified columns
        # print(f"Shape of df: {X.shape}")
        X.drop(labels=self.columns_to_drop, axis = 1, inplace = True)
        # print(f"Shape of df after dropping columns: {X.shape}")

        return X