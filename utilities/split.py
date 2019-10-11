def time_split(X, test_size=0.1):
    """Splits X in time if it is chronologically sorted.

    :param X: 
    :param test_size: 
    :returns: X, X_test
    :rtype: 

    """
    split_index = int(len(X)*(1 - test_size))
    return X[:split_index], X[split_index:]

