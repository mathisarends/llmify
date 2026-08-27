import httpx
import pytest

pytest.importorskip("google.genai")

from google.genai import errors

from llmify.exceptions import (
    AuthenticationError,
    ContextLengthExceededError,
    OutOfCreditsError,
    RateLimitError,
    RetryableError,
)
from llmify.providers.google.client import _map_google_error


def _api_error(
    code: int, message: str = "boom", response: httpx.Response | None = None
) -> errors.APIError:
    return errors.APIError(code, {"message": message}, response=response)


def _response(status_code: int, **headers: str) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        headers=headers,
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
    )


class TestStatusCodeMapping:
    @pytest.mark.parametrize(
        ("status_code", "expected"),
        [
            (401, AuthenticationError),
            (403, AuthenticationError),
            (402, OutOfCreditsError),
            (429, RateLimitError),
            (408, RetryableError),
            (409, RetryableError),
            (500, RetryableError),
            (502, RetryableError),
            (503, RetryableError),
        ],
    )
    def test_maps_status_code_to_llmify_error(
        self, status_code: int, expected: type[Exception]
    ) -> None:
        assert isinstance(_map_google_error(_api_error(status_code)), expected)

    @pytest.mark.parametrize("status_code", [500, 503])
    def test_retryable_carries_the_status_code(self, status_code: int) -> None:
        mapped = _map_google_error(_api_error(status_code))
        assert isinstance(mapped, RetryableError)
        assert mapped.status_code == status_code

    @pytest.mark.parametrize("status_code", [400, 404, 422])
    def test_passes_through_non_retryable_client_errors(self, status_code: int) -> None:
        error = _api_error(status_code, "invalid argument")
        assert _map_google_error(error) is error


class TestContextLengthDetection:
    @pytest.mark.parametrize(
        "message",
        [
            "The input context is too large",
            "Input token count exceeds the maximum",
            "token limit exceeded for this model",
            "request exceeds the maximum number of tokens",
        ],
    )
    def test_detects_context_length_from_message(self, message: str) -> None:
        mapped = _map_google_error(_api_error(400, message))
        assert isinstance(mapped, ContextLengthExceededError)

    @pytest.mark.parametrize(
        "message",
        [
            "invalid argument",
            "unsupported token",  # 'token' alone is not enough
            "the maximum number of candidates is 8",  # no 'token' or 'context'
        ],
    )
    def test_leaves_unrelated_400s_alone(self, message: str) -> None:
        error = _api_error(400, message)
        assert _map_google_error(error) is error

    def test_only_applies_to_400(self) -> None:
        mapped = _map_google_error(_api_error(500, "context too large"))
        assert isinstance(mapped, RetryableError)


class TestRetryAfter:
    def test_parses_the_retry_after_header(self) -> None:
        mapped = _map_google_error(
            _api_error(429, response=_response(429, **{"retry-after": "2.5"}))
        )
        assert isinstance(mapped, RateLimitError)
        assert mapped.retry_after == 2.5

    def test_defaults_to_none_without_a_response(self) -> None:
        mapped = _map_google_error(_api_error(429))
        assert isinstance(mapped, RateLimitError)
        assert mapped.retry_after is None

    def test_defaults_to_none_without_the_header(self) -> None:
        mapped = _map_google_error(_api_error(429, response=_response(429)))
        assert isinstance(mapped, RateLimitError)
        assert mapped.retry_after is None

    def test_ignores_an_unparseable_header(self) -> None:
        # Google may send an HTTP-date instead of a number; that must not raise.
        mapped = _map_google_error(
            _api_error(
                429,
                response=_response(
                    429, **{"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}
                ),
            )
        )
        assert isinstance(mapped, RateLimitError)
        assert mapped.retry_after is None


class TestNonApiErrors:
    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectError("connection refused"),
            httpx.ReadTimeout("timed out"),
            httpx.RemoteProtocolError("peer closed connection"),
        ],
    )
    def test_maps_transport_errors_to_retryable(self, exc: Exception) -> None:
        assert isinstance(_map_google_error(exc), RetryableError)

    def test_passes_through_unrelated_exceptions(self) -> None:
        exc = ValueError("something else entirely")
        assert _map_google_error(exc) is exc
