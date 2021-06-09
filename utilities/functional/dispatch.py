from functools import singledispatch
import pandas as pd
import numpy as np


@singledispatch
def count(arg):
    pass


@count.register
def _(df: pd.DataFrame):
    return df.shape[0] * df.shape[1]


@count.register
def _(l: list):
    return len(l)


assert count(pd.DataFrame(np.random.randn(10, 10))) == 100
assert count(list(range(10))) == 10
