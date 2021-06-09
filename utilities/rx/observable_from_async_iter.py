import asyncio
import logging
from typing import AsyncIterator, Callable, Optional, Any

import rx
from rx.core import Observable
from rx.core.typing import Observer, Scheduler, Mapper
from rx.disposable import Disposable
from rx.subject import Subject


def observable_from_aiter(iter: AsyncIterator, loop: asyncio.AbstractEventLoop):
    def on_subscribe(observer: Observer, scheduler: Scheduler):
        async def _aio_sub():
            async for i in iter:
                observer.on_next(i)

        task = loop.create_task(_aio_sub())
        return Disposable(lambda: task.cancel())

    return rx.create(on_subscribe)


def subject(s: Subject) -> Callable[[Observable], Observable]:
    def _subject(source: Observable) -> Observable:
        source.subscribe(
            s, on_error=lambda e: logging.error(f"Got error:{e} from subject:{s}")
        )
        return s

    return _subject


def flatter(mapper: Optional[Mapper] = None) -> Callable[[Observable], Observable]:
    mapper = mapper if mapper else lambda x: x

    def _flatter(source: Observable) -> Observable:
        def subscribe(obv: Observer, scheduler: Scheduler = None) -> Disposable:
            def on_next(value: Any) -> None:
                [obv.on_next(v) for v in mapper(value)]

            return source.subscribe_(on_next, obv.on_error, obv.on_completed, scheduler)

        return Observable(subscribe)

    return _flatter
