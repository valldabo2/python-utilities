# https://toolz.readthedocs.io/en/latest/api.html#toolz.functoolz.compose

from toolz.functoolz import compose
from functools import partial


def add(x, y):
    return x + y


add_one = partial(add, 1)
f = compose(str, add_one)
assert f(1) == "2"
