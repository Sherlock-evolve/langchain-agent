"""Small bridges that preserve generator send/close semantics across asyncio."""

from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator, Callable, Generator, Iterator
from threading import Thread


_CLOSE = object()


async def bridge_sync_generator(
    generator_factory: Callable[[], Generator],
    *,
    cancel_callback: Callable[[], object] | None = None,
):
    """Expose a synchronous two-way generator as an asynchronous generator.

    Each async ``asend(value)`` becomes ``send(value)`` on the worker-owned
    generator. Closing or cancelling the async consumer first invokes the
    cancellation callback, then closes the synchronous generator at its next
    safe yield boundary.
    """

    if not callable(generator_factory):
        raise TypeError("generator_factory must be callable")
    if cancel_callback is not None and not callable(cancel_callback):
        raise TypeError("cancel_callback must be callable")

    outputs: queue.Queue[tuple[str, object]] = queue.Queue()
    commands: queue.Queue[object] = queue.Queue()

    def publish(status: str, value: object) -> None:
        outputs.put((status, value))

    def run() -> None:
        stream = None
        completed = False
        decision = None
        try:
            stream = generator_factory()
            while True:
                try:
                    item = stream.send(decision)
                except StopIteration:
                    completed = True
                    publish("done", None)
                    return
                publish("item", item)
                command = commands.get()
                if command is _CLOSE:
                    return
                decision = command
        except BaseException as error:
            publish("error", error)
        finally:
            if stream is not None and not completed:
                try:
                    stream.close()
                except BaseException:
                    pass

    worker = Thread(
        target=run,
        name="workspace-agent-async-stream",
        daemon=True,
    )
    worker.start()
    completed = False
    try:
        while True:
            while True:
                try:
                    status, value = outputs.get_nowait()
                    break
                except queue.Empty:
                    await asyncio.sleep(0.001)
            if status == "done":
                completed = True
                return
            if status == "error":
                completed = True
                if isinstance(value, BaseException):
                    raise value
                raise RuntimeError("async bridge received an invalid error")
            if status != "item":
                raise RuntimeError("async bridge received an invalid outcome")
            decision = yield value
            commands.put(decision)
    finally:
        if not completed:
            if cancel_callback is not None:
                cancel_callback()
            commands.put(_CLOSE)
        worker.join(timeout=0.5)


def iterate_async_synchronously(
    async_iterator: AsyncIterator,
) -> Iterator:
    """Drive an async iterator from a worker thread with live item delivery."""

    if not hasattr(async_iterator, "__aiter__"):
        raise TypeError("value is not an async iterator")
    loop = asyncio.new_event_loop()
    iterator = async_iterator.__aiter__()
    try:
        while True:
            try:
                item = loop.run_until_complete(iterator.__anext__())
            except StopAsyncIteration:
                return
            yield item
    finally:
        close = getattr(iterator, "aclose", None)
        if callable(close):
            try:
                loop.run_until_complete(close())
            except BaseException:
                pass
        loop.close()


def run_coroutine_synchronously(awaitable):
    """Run one awaitable in a thread that has no active event loop."""

    return asyncio.run(awaitable)


async def run_sync_asynchronously(
    function: Callable,
    *args,
    **kwargs,
):
    """Run blocking work without using asyncio's process-wide executor."""

    if not callable(function):
        raise TypeError("function must be callable")
    outcomes: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            outcomes.put((True, function(*args, **kwargs)))
        except BaseException as error:
            outcomes.put((False, error))

    worker = Thread(
        target=run,
        name="workspace-agent-async-call",
        daemon=True,
    )
    worker.start()
    while True:
        try:
            succeeded, value = outcomes.get_nowait()
            break
        except queue.Empty:
            await asyncio.sleep(0.001)
    worker.join(timeout=0.1)
    if succeeded:
        return value
    if isinstance(value, BaseException):
        raise value
    raise RuntimeError("async call bridge received an invalid outcome")
