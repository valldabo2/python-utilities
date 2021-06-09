from joblib import Parallel, delayed


def square(x):
    return x * x


assert Parallel()(delayed(square)(x) for x in range(1, 3)) == [1, 4]
