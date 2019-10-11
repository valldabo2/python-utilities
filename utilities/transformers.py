import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class ColumnSelectorPandas(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.is_series_and_name_match = False

    def fit(self, X, y=None):
        existing_columns = []
        if type(X) == pd.DataFrame:
            data_cols = X.columns
        elif type(X) == pd.Series:
            if X.name in self.columns:
                self.columns = []
                self.is_series_and_name_match = True

        else:
            pass
            # print(f'Not implemented for type {type(X)}')

        for col in self.columns:
            if col not in data_cols:
                # print(f'Col:{col} not in data columns')
                pass
            else:
                existing_columns.append(col)
        self.existing_columns = existing_columns

        return self

    def transform(self, X, y=None, copy=None):
        if self.is_series_and_name_match:
            return X.to_frame(X.name)
        elif len(self.existing_columns) == 0:
            pass
        else:
            return X.loc[:, self.existing_columns]


class LabelEncoderPandas(BaseEstimator, TransformerMixin):
    def __init__(self, fill_na_value=-1):
        self.labels = {}  # {col: {val:label, val2: label2}}
        self.fill_na_value = fill_na_value

    def fit(self, X, y=None):
        for col in X.columns:
            self.labels[col] = {}
            unique_values = X[col].fillna(self.fill_na_value).unique()
            for label, value in enumerate(unique_values):
                self.labels[col][value] = label
        return self

    def transform(self, X, y=None):
        labeled_cols = list(self.labels.keys())

        for col in labeled_cols:

            if self.fill_na_value is not None:
                X[col] = X[col].fillna(self.fill_na_value)

            X[col] = X[col].map(
                self.labels[col],
                na_action='ignore')  # Will give na if not in labels

        X[labeled_cols] = X[labeled_cols].fillna(self.fill_na_value)
        X[labeled_cols] = X[labeled_cols].astype(int)

        return X


class FilterNanPandas(BaseEstimator, TransformerMixin):
    def __init__(self, max_pct_nans):
        self.max_pct_nans = max_pct_nans

    def fit(self, X, y=None):
        n_nans = X.isnull().sum(axis=0)
        pct_nans = n_nans / len(X)
        ok_columns = pct_nans[pct_nans <= self.max_pct_nans].index
        self.ok_columns = ok_columns
        return self

    def transform(self, X, y=None):
        cols = [col for col in X.columns if col in self.ok_columns]
        return X.loc[:, cols]


def impute_mean(df):
    return df.fillna(df.mean(axis=0))


def impute_median(df):
    return df.fillna(df.median(axis=0))


def impute_most_common(df):
    if type(df) == pd.DataFrame:
        for col in df:
            most_common_value = df[col].value_counts().sort_values()
            df[col] = df[col].fillna(most_common_value)
    else:
        most_common_value = df.value_counts().sort_values()
        df = df.fillna(most_common_value)
    return df


def clean_inf_nan(df):
    return df.replace([np.inf, -np.inf], np.nan)


class MapColumnsPandas(BaseEstimator, TransformerMixin):
    def __init__(self, column_map):
        self.column_map = column_map
        self.is_series = False
        self.incorrect_series_name = False

    def fit(self, X, y=None):

        if type(X) == pd.DataFrame:
            data_cols = X.columns
            column_map = {
                col: map_
                for col, map_ in self.column_map.items() if col in data_cols
            }
            self.existing_col_map = column_map
        elif type(X) == pd.Series:
            if X.name in self.column_map:
                self.series_map = self.column_map[X.name]
                self.is_series = True
            else:
                self.incorrect_series_name = True
        return self

    def transform(self, X, y=None):
        if self.is_series:
            return X.map(self.series_map).to_frame()
        elif self.incorrect_series_name:
            return X.to_frame()
        else:
            data_to_return = []
            for col, map_ in self.existing_col_map.items():
                mapped_column = 'mapped_' + col
                data_to_return.append(X[col].map(map_).rename(mapped_column))
            return pd.concat(data_to_return, axis=1)


class ColumnFilterPandas(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_filter):
        self.columns_to_filter = columns_to_filter

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, copy=None):
        columns = [
            col for col in X.columns if col not in self.columns_to_filter
        ]
        return X[columns]
