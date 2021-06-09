def sessionize(timestamp, time_delta):
    """ Sessionizes based on timestamp and timedelta"""
    return (timestamp - timestamp.shift(1) > time_delta).cumsum() + 1
