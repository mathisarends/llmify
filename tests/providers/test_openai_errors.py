from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("openai")

import httpx
from openai import (
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
from llmify.providers.openai_compatible import OpenAICompatible, _map_openai_error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_response(status_code: int, json_body: dict) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=json_body,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )


def _rate_limit_error(body: dict | None = None) -> RateLimitError:
    response = _make_openai_response(429, body or {})
    return RateLimitError(response=response, body=body or {}, message="rate limited")


def _api_status_error(status_code: int, body: dict | None = None) -> APIStatusError:
    response = _make_openai_response(status_code, body or {})
    return APIStatusError(message="server error", response=response, body=body or {})


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/"))


def _timeout_error() -> APITimeoutError:
    return APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/"))


class MockChatModel(OpenAICompatible):
    def __init__(self):
        ChatModel.__init__(self, model="gpt-4")
        self._client = AsyncMock()


# ---------------------------------------------------------------------------
# _map_openai_error unit tests
# ---------------------------------------------------------------------------


class TestMapOpenAIError:
    def test_rate_limit_maps_to_llm_rate_limit(self) -> None:
        exc = _rate_limit_error()
        result = _map_openai_error(exc)
        assert isinstance(result, LLMRateLimitError)
        assert result.status_code == 429

    def test_rate_limit_with_insufficient_quota_maps_to_out_of_credits(self) -> None:
        body = {"error": {"code": "insufficient_quota", "message": "quota exceeded"}}
        exc = _rate_limit_error(body)
        result = _map_openai_error(exc)
        assert isinstance(result, OutOfCreditsError)

    def test_rate_limit_parses_retry_after_header(self) -> None:
        response = _make_openai_response(429, {})
        # Rebuild with retry-after header
        response = httpx.Response(
            status_code=429,
            headers={"retry-after": "30"},
            json={},
            request=httpx.Request("POST", "https://api.openai.com/"),
        )
        exc = RateLimitError(response=response, body={}, message="rate limited")
        result = _map_openai_error(exc)
        assert isinstance(result, LLMRateLimitError)
        assert result.retry_after == 30.0

    def test_connection_error_maps_to_retryable(self) -> None:
        result = _map_openai_error(_connection_error())
        assert isinstance(result, RetryableError)

    def test_timeout_error_maps_to_retryable(self) -> None:
        result = _map_openai_error(_timeout_error())
        assert isinstance(result, RetryableError)

    def test_500_status_maps_to_retryable(self) -> None:
        result = _map_openai_error(_api_status_error(500))
        assert isinstance(result, RetryableError)
        assert result.status_code == 500

    def test_503_status_maps_to_retryable(self) -> None:
        result = _map_openai_error(_api_status_error(503))
        assert isinstance(result, RetryableError)

    def test_400_status_passes_through_unchanged(self) -> None:
        exc = _api_status_error(400)
        result = _map_openai_error(exc)
        assert result is exc

    def test_context_length_exceeded_maps_correctly(self) -> None:
        body = {
            "error": {"code": "context_length_exceeded", "message": "too many tokens"}
        }
        exc = _api_status_error(400, body)
        result = _map_openai_error(exc)
        assert isinstance(result, ContextLengthExceededError)

    def test_401_status_maps_to_authentication_error(self) -> None:
        result = _map_openai_error(_api_status_error(401))
        assert isinstance(result, LLMAuthenticationError)

    def test_unknown_exception_passes_through_unchanged(self) -> None:
        exc = ValueError("something else")
        assert _map_openai_error(exc) is exc


# ---------------------------------------------------------------------------
# Integration: invoke raises correct llmify exceptions
# ---------------------------------------------------------------------------


class TestInvokeErrorMapping:
    @pytest.mark.asyncio
    async def test_invoke_raises_rate_limit_error(self) -> None:
        model = MockChatModel()
        model._client.chat.completions.create = AsyncMock(
            side_effect=_rate_limit_error()
        )
        with pytest.raises(LLMRateLimitError):
            await model.invoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_invoke_raises_out_of_credits(self) -> None:
        model = MockChatModel()
        body = {"error": {"code": "insufficient_quota"}}
        model._client.chat.completions.create = AsyncMock(
            side_effect=_rate_limit_error(body)
        )
        with pytest.raises(OutOfCreditsError):
            await model.invoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_invoke_raises_retryable_on_connection_error(self) -> None:
        model = MockChatModel()
        model._client.chat.completions.create = AsyncMock(
            side_effect=_connection_error()
        )
        with pytest.raises(RetryableError):
            await model.invoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_invoke_raises_retryable_on_500(self) -> None:
        model = MockChatModel()
        model._client.chat.completions.create = AsyncMock(
            side_effect=_api_status_error(500)
        )
        with pytest.raises(RetryableError):
            await model.invoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_invoke_preserves_unrelated_errors(self) -> None:
        model = MockChatModel()
        model._client.chat.completions.create = AsyncMock(
            side_effect=_api_status_error(400)
        )
        with pytest.raises(APIStatusError):
            await model.invoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_invoke_raises_context_length_exceeded(self) -> None:
        model = MockChatModel()
        body = {"error": {"code": "context_length_exceeded"}}
        model._client.chat.completions.create = AsyncMock(
            side_effect=_api_status_error(400, body)
        )
        with pytest.raises(ContextLengthExceededError):
            await model.invoke([UserMessage(content="hi")])

    @pytest.mark.asyncio
    async def test_invoke_raises_authentication_error(self) -> None:
        model = MockChatModel()
        model._client.chat.completions.create = AsyncMock(
            side_effect=_api_status_error(401)
        )
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
    async def test_stream_raises_rate_limit_on_create(self) -> None:
        model = MockChatModel()
        model._client.chat.completions.create = AsyncMock(
            side_effect=_rate_limit_error()
        )
        with pytest.raises(LLMRateLimitError):
            await _collect(model.stream([UserMessage(content="hi")]))

    @pytest.mark.asyncio
    async def test_stream_raises_retryable_on_connection_error_during_iteration(
        self,
    ) -> None:
        model = MockChatModel()

        async def _failing_stream():
            yield SimpleNamespace(usage=None, choices=[])
            raise _connection_error()

        mock_stream = _failing_stream()
        model._client.chat.completions.create = AsyncMock(return_value=mock_stream)

        with pytest.raises(RetryableError):
            await _collect(model.stream([UserMessage(content="hi")]))

    @pytest.mark.asyncio
    async def test_stream_raises_out_of_credits_on_create(self) -> None:
        model = MockChatModel()
        body = {"error": {"code": "insufficient_quota"}}
        model._client.chat.completions.create = AsyncMock(
            side_effect=_rate_limit_error(body)
        )
        with pytest.raises(OutOfCreditsError):
            await _collect(model.stream([UserMessage(content="hi")]))
