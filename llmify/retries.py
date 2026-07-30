import asyncio
import inspect
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from llmify.exceptions import RateLimitError, RetryableError


@dataclass(frozen=True, slots=True)
class RetryEvent:
    retry_number: int
    max_retries: int
    delay: float
    error: RetryableError

    @property
    def failed_attempt(self) -> int:
        return self.retry_number

    @property
    def next_attempt(self) -> int:
        return self.retry_number + 1

    @property
    def max_attempts(self) -> int:
        return self.max_retries + 1


type RetryCallback = Callable[[RetryEvent], Awaitable[None] | None]


def retry_delay(error: RetryableError, retry_number: int) -> float:
    if isinstance(error, RateLimitError) and error.retry_after is not None:
        return max(error.retry_after, 0.0)

    exponential_delay = min(0.5 * 2**retry_number, 8.0)
    return exponential_delay * random.uniform(0.75, 1.0)


async def sleep_before_retry(
    error: RetryableError,
    retry_number: int,
    max_retries: int,
    on_retry: RetryCallback | None,
) -> None:
    delay = retry_delay(error, retry_number)
    event = RetryEvent(
        retry_number=retry_number + 1,
        max_retries=max_retries,
        delay=delay,
        error=error,
    )

    if on_retry is not None:
        result = on_retry(event)
        if inspect.isawaitable(result):
            await result

    await asyncio.sleep(delay)
