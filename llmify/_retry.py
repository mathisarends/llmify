import asyncio
import random

from llmify.exceptions import RateLimitError, RetryableError


def retry_delay(error: RetryableError, retry_number: int) -> float:
    if isinstance(error, RateLimitError) and error.retry_after is not None:
        return max(error.retry_after, 0.0)

    exponential_delay = min(0.5 * 2**retry_number, 8.0)
    return exponential_delay * random.uniform(0.75, 1.0)


async def sleep_before_retry(error: RetryableError, retry_number: int) -> None:
    await asyncio.sleep(retry_delay(error, retry_number))
