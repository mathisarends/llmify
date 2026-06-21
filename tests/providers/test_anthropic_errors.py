from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("anthropic")

import httpx
from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)

from llmify.base import ChatModel
from llmify.exceptions import OutOfCreditsError
from llmify.exceptions import AuthenticationError as LLMAuthenticationError
from llmify.exceptions import ContextLengthExceededError
from llmify.exceptions import RateLimitError as LLMRateLimitError
from llmify.exceptions import RetryableError
from llmify.messages import UserMessage
from llmify.providers.anthropic import ChatAnthropic, _map_anthropic_error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_anthropic_response(status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json={},
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )


def _rate_limit_error(body: dict | None = None) -> RateLimitError:
    response = _make_anthropic_response(429)
    return RateLimitError(response=response, body=body or {}, message="rate limited")


def _api_status_error(status_code: int, body: dict | None = None) -> APIStatusError:
    response = _make_anthropic_response(status_code)
    return APIStatusError(message="error", response=response, body=body or {})


def _connection_error() -> APIConnectionError:
    return APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/")
    )


def _timeout_error() -> APITimeoutError:
    return APITimeoutError(request=httpx.Request("POST", "https://api.anthropic.com/"))


class MockAnthropicModel(ChatAnthropic):
    def __init__(self):
        ChatModel.__init__(self, model="claude-test")
        self._client = SimpleNamespace(messages=SimpleNamespace())


# ---------------------------------------------------------------------------
# _map_anthropic_error unit tests
# ---------------------------------------------------------------------------


class TestMapAnthropicError:
    def test_rate_limit_maps_to_llm_rate_limit(self) -> None:
        result = _map_anthropic_error(_rate_limit_error())
        assert isinstance(result, LLMRateLimitError)
        assert result.status_code == 429

    def test_rate_limit_with_credit_balance_too_low_maps_to_out_of_credits(
        self,
    ) -> None:
        body = {"error": {"type": "credit_balance_too_low"}}
        result = _map_anthropic_error(_rate_limit_error(body))
        assert isinstance(result, OutOfCreditsError)

    def test_402_status_maps_to_out_of_credits(self) -> None:
        result = _map_anthropic_error(_api_status_error(402))
        assert isinstance(result, OutOfCreditsError)

    def test_rate_limit_parses_retry_after_header(self) -> None:
        response = httpx.Response(
            status_code=429,
            headers={"retry-after": "60"},
            json={},
            request=httpx.Request("POST", "https://api.anthropic.com/"),
        )
        exc = RateLimitError(response=response, body={}, message="rate limited")
        result = _map_anthropic_error(exc)
        assert isinstance(result, LLMRateLimitError)
        assert result.retry_after == 60.0

    def test_connection_error_maps_to_retryable(self) -> None:
        result = _map_anthropic_error(_connection_error())
        assert isinstance(result, RetryableError)

    def test_timeout_error_maps_to_retryable(self) -> None:
        result = _map_anthropic_error(_timeout_error())
        assert isinstance(result, RetryableError)

    def test_500_status_maps_to_retryable(self) -> None:
        result = _map_anthropic_error(_api_status_error(500))
        assert isinstance(result, RetryableError)
        assert result.status_code == 500

    def test_529_status_maps_to_retryable(self) -> None:
        result = _map_anthropic_error(_api_status_error(529))
        assert isinstance(result, RetryableError)

    def test_400_status_passes_through_unchanged(self) -> None:
        exc = _api_status_error(400)
        assert _map_anthropic_error(exc) is exc

    def test_context_length_exceeded_maps_correctly(self) -> None:
        body = {
            "error": {
                "type": "invalid_request_error",
                "message": "prompt is too long: 273323 tokens > 200000 maximum",
            }
        }
        exc = _api_status_error(400, body)
        result = _map_anthropic_error(exc)
        assert isinstance(result, ContextLengthExceededError)

    def test_context_length_from_token_exceeded_message(self) -> None:
        body = {
            "error": {
                "type": "invalid_request_error",
                "message": "Input token count exceeds the limit of 100000",
            }
        }
        exc = _api_status_error(400, body)
        result = _map_anthropic_error(exc)
        assert isinstance(result, ContextLengthExceededError)

    def test_401_status_maps_to_authentication_error(self) -> None:
        result = _map_anthropic_error(_api_status_error(401))
        assert isinstance(result, LLMAuthenticationError)

    def test_unknown_exception_passes_through_unchanged(self) -> None:
        exc = ValueError("something else")
        assert _map_anthropic_error(exc) is exc


# ---------------------------------------------------------------------------
# Integration: invoke raises correct llmify exceptions
# ---------------------------------------------------------------------------


class FakeAsyncContext:
    """Minimal async context manager that raises on entry."""

    def __init__(self, exc: Exception):
        self._exc = exc

    async def __aenter__(self):
        raise self._exc

    async def __aexit__(self, *_):
        return None


class TestInvokeErrorMapping:
    @pytest.mark.asyncio
    async def test_invoke_raises_rate_limit_error(self) -> None:
        model = MockAnthropicModel()
        model._client.messages.create = AsyncMock(side_effect=_rate_limit_error())
        with pytest.raises(LLMRateLimitError):
            await model.invoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_invoke_raises_out_of_credits_on_402(self) -> None:
        model = MockAnthropicModel()
        model._client.messages.create = AsyncMock(side_effect=_api_status_error(402))
        with pytest.raises(OutOfCreditsError):
            await model.invoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_invoke_raises_out_of_credits_on_credit_balance_too_low(self) -> None:
        model = MockAnthropicModel()
        body = {"error": {"type": "credit_balance_too_low"}}
        model._client.messages.create = AsyncMock(side_effect=_rate_limit_error(body))
        with pytest.raises(OutOfCreditsError):
            await model.invoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_invoke_raises_retryable_on_connection_error(self) -> None:
        model = MockAnthropicModel()
        model._client.messages.create = AsyncMock(side_effect=_connection_error())
        with pytest.raises(RetryableError):
            await model.invoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_invoke_raises_retryable_on_500(self) -> None:
        model = MockAnthropicModel()
        model._client.messages.create = AsyncMock(side_effect=_api_status_error(500))
        with pytest.raises(RetryableError):
            await model.invoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_invoke_preserves_unrelated_errors(self) -> None:
        model = MockAnthropicModel()
        model._client.messages.create = AsyncMock(side_effect=_api_status_error(400))
        with pytest.raises(APIStatusError):
            await model.invoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_invoke_raises_context_length_exceeded(self) -> None:
        model = MockAnthropicModel()
        body = {
            "error": {
                "type": "invalid_request_error",
                "message": "prompt is too long: 300000 tokens > 200000 maximum",
            }
        }
        model._client.messages.create = AsyncMock(
            side_effect=_api_status_error(400, body)
        )
        with pytest.raises(ContextLengthExceededError):
            await model.invoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_invoke_raises_authentication_error(self) -> None:
        model = MockAnthropicModel()
        model._client.messages.create = AsyncMock(side_effect=_api_status_error(401))
        with pytest.raises(LLMAuthenticationError):
            await model.invoke([UserMessage(content="hi")])


# ---------------------------------------------------------------------------
# Integration: stream raises correct llmify exceptions
# ---------------------------------------------------------------------------


async def _collect(gen):
    events = []
    async for event in gen:
        events.append(event)
    return events


class TestStreamErrorMapping:
    @pytest.mark.asyncio
    async def test_stream_raises_rate_limit_on_stream_open(self) -> None:
        model = MockAnthropicModel()
        model._client.messages.stream = lambda **_: FakeAsyncContext(
            _rate_limit_error()
        )
        with pytest.raises(LLMRateLimitError):
            await _collect(model.stream([UserMessage(content="hi")]))

    @pytest.mark.asyncio
    async def test_stream_raises_out_of_credits_on_402(self) -> None:
        model = MockAnthropicModel()
        model._client.messages.stream = lambda **_: FakeAsyncContext(
            _api_status_error(402)
        )
        with pytest.raises(OutOfCreditsError):
            await _collect(model.stream([UserMessage(content="hi")]))

    @pytest.mark.asyncio
    async def test_stream_raises_retryable_on_connection_error(self) -> None:
        model = MockAnthropicModel()
        model._client.messages.stream = lambda **_: FakeAsyncContext(
            _connection_error()
        )
        with pytest.raises(RetryableError):
            await _collect(model.stream([UserMessage(content="hi")]))
