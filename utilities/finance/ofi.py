def i(b):
    return b.astype(int)


def ofi(b, b_v, a, a_v):
    """ Calculates the Order Flow Imbalance at timepoint t
    :param b: Bids
    :param b_v: Bid volumes
    :param a: Asks
    :param a_v: Ask volumes
    :return: Order Flow Imbalance
    :rtype: pd.Series
    """
    b_, b_v_, a_, a_v_ = b.shift(1), b_v.shift(1), a.shift(1), a_v.shift(1)
    ofi = i(b >= b_) * b_v - i(b <= b_) * b_v_ - i(a <= a_) * a_v + i(a >= a_) * a_v_
    return ofi.fillna(0)
