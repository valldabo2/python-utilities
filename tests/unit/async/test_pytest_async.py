import asyncio
import pytest


async def return_2():
    return 2


@pytest.mark.asyncio
async def test_some_asyncio_code():
    res = await return_2()
    assert res == 2
