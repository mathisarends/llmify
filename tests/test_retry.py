import pytest

from llmify._retry import retry_delay
from llmify.exceptions import RateLimitError, RetryableError


def test_uses_retry_after_from_rate_limit() -> None:
    assert retry_delay(RateLimitError(retry_after=3.5), 0) == 3.5


def test_negative_retry_after_is_immediate() -> None:
    assert retry_delay(RateLimitError(retry_after=-1), 0) == 0.0


@pytest.mark.parametrize(
    ("retry_number", "expected"),
    [(0, 0.5), (1, 1.0), (4, 8.0), (10, 8.0)],
)
def test_uses_capped_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
    retry_number: int,
    expected: float,
) -> None:
    monkeypatch.setattr("llmify._retry.random.uniform", lambda _start, _end: 1.0)

    assert retry_delay(RetryableError("transient"), retry_number) == expected
