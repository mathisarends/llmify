import functools
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, cast


class _Decorator(Protocol):
    def __call__[**P, R](
        self,
        func: Callable[P, R],
        /,
    ) -> Callable[P, R]: ...


def timed(
    additional_text: str = "",
    min_duration_to_log: float = 0.25,
) -> _Decorator:
    def decorator[**P, R](func: Callable[P, R]) -> Callable[P, R]:
        function_name = additional_text.strip("-") or getattr(
            func, "__name__", type(func).__name__
        )

        def log_duration(start_time: float) -> None:
            execution_time = time.perf_counter() - start_time
            if execution_time > min_duration_to_log:
                logger = logging.getLogger(func.__module__)
                logger.debug("⏳ %s() took %.2fs", function_name, execution_time)

        if inspect.iscoroutinefunction(func):
            async_func = cast(Callable[P, Awaitable[Any]], func)

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                start_time = time.perf_counter()
                result = await async_func(*args, **kwargs)
                log_duration(start_time)
                return result

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            log_duration(start_time)
            return result

        return wrapper

    return decorator
