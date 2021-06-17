import asyncio


async def sleep(time):
    await asyncio.sleep(time)


async def sleep_print(time, text):
    await sleep(time)
    print(text)


loop = asyncio.get_event_loop()
t1 = loop.create_task(sleep_print(0.1, "Hi"))
t2 = loop.create_task(sleep_print(0.2, "Bye"))
loop.run_until_complete(sleep(0.3))
