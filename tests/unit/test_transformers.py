import pandas as pd
import numpy as np
from utilities.transformers import ColumnSelectorPandas
from utilities.transformers import LabelEncoderPandas
from utilities.transformers import FilterNanPandas
from pandas.testing import assert_frame_equal


def print_df_str(df):
    print('\n' + df.to_string())


def test_column_selector():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    transformer = ColumnSelectorPandas(['a'])
    df_ = transformer.fit_transform(df)
    assert list(df_.columns) == ['a'], 'Incorrect columns'


def test_label_encoder():
    df = pd.DataFrame({'a': [1, 2, 1], 'b': [3, 4, np.nan]})
    transformer = LabelEncoderPandas()
    df_ = transformer.fit_transform(df)
    expected_df = pd.DataFrame({'a': [0, 1, 0], 'b': [0, 1, 2]})
    assert_frame_equal(expected_df, df_)


def test_filter_nan():
    df = pd.DataFrame({'a': [1, 2, 3, np.nan, np.nan], 'b': [3, 4, 5, 6, np.nan]})
    transformer = FilterNanPandas(max_pct_nans=0.3)
    df_ = transformer.fit_transform(df)
    assert_frame_equal(df['b'].to_frame(), df_)
