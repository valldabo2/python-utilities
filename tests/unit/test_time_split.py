from utilities.split import time_split

def test_time_split():
    X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    test_size = 0.2
    X, X_test = time_split(X, test_size=test_size)
    assert X_test == [9, 10], "Did not split data correctly"

