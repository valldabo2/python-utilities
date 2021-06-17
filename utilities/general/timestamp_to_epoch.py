from pytz import UTC
from datetime import datetime

EPOCH = UTC.localize(datetime.utcfromtimestamp(0))


def timestamp_to_nano_epoch(timestamp: datetime) -> int:
    if not timestamp.tzinfo:
        timestamp = UTC.localize(timestamp)
    # Assumes NS precision always.
    return int((timestamp - EPOCH).total_seconds() * 1e9)
