import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("google.genai")

from google.genai import errors, types

from llmify import retries as retries_provider
from llmify.exceptions import RetryableError
from llmify.messages import UserMessage
from llmify.providers.google import ChatGoogle
from llmify.views import StreamEnd, StreamToolCall


class FakeChunkStream:
    """Stands in for the async iterator returned by generate_content_stream."""

    def __init__(self, chunks: list[Any], raise_after: Exception | None = None):
        self._chunks = chunks
        self._raise_after = raise_after

    def __aiter__(self):
        async def generator():
            for chunk in self._chunks:
                yield chunk
            if self._raise_after is not None:
                raise self._raise_after

        return generator()


def _model(**kwargs: Any) -> ChatGoogle:
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace()))
    kwargs.setdefault("max_retries", 0)
    return ChatGoogle(model="gemini-test", client=client, **kwargs)


def _chunk(
    *parts: types.Part,
    finish_reason: str | None = None,
    usage: types.GenerateContentResponseUsageMetadata | None = None,
) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=list(parts)),
                finish_reason=finish_reason,
            )
        ],
        usage_metadata=usage,
    )


async def _collect(stream) -> list[Any]:
    return [event async for event in stream]


class TestGoogleStreaming:
    @pytest.mark.asyncio
    async def test_emits_text_deltas_and_a_final_completion(self) -> None:
        model = _model()
        model._client.models.generate_content_stream = AsyncMock(
            return_value=FakeChunkStream(
                [
                    _chunk(types.Part(text="Hello ")),
                    _chunk(types.Part(text="world"), finish_reason="STOP"),
                ]
            )
        )

        events = await _collect(model.stream([UserMessage(content="Hi")]))

        assert [event.type for event in events] == ["text", "text", "end"]
        assert [event.delta for event in events[:2]] == ["Hello ", "world"]

        end = events[-1]
        assert isinstance(end, StreamEnd)
        assert end.completion == "Hello world"
        assert end.stop_reason == "STOP"

    @pytest.mark.asyncio
    async def test_skips_empty_text_chunks(self) -> None:
        model = _model()
        model._client.models.generate_content_stream = AsyncMock(
            return_value=FakeChunkStream(
                [_chunk(), _chunk(types.Part(text="Hi")), _chunk()]
            )
        )

        events = await _collect(model.stream([UserMessage(content="Hi")]))

        assert [event.type for event in events] == ["text", "end"]

    @pytest.mark.asyncio
    async def test_emits_tool_calls_and_repeats_them_at_the_end(self) -> None:
        model = _model()
        model._client.models.generate_content_stream = AsyncMock(
            return_value=FakeChunkStream(
                [
                    _chunk(
                        types.Part(
                            function_call=types.FunctionCall(
                                id="call_1",
                                name="get_weather",
                                args={"city": "Berlin"},
                            )
                        ),
                        finish_reason="STOP",
                    )
                ]
            )
        )

        events = await _collect(model.stream([UserMessage(content="Weather?")]))

        assert [event.type for event in events] == ["tool_call", "end"]
        streamed = events[0]
        assert isinstance(streamed, StreamToolCall)
        assert streamed.tool_call.function.name == "get_weather"
        assert json.loads(streamed.tool_call.function.arguments) == {"city": "Berlin"}

        end = events[-1]
        assert isinstance(end, StreamEnd)
        assert [call.id for call in end.tool_calls] == ["call_1"]

    @pytest.mark.asyncio
    async def test_keeps_the_last_reported_usage(self) -> None:
        model = _model()
        model._client.models.generate_content_stream = AsyncMock(
            return_value=FakeChunkStream(
                [
                    _chunk(
                        types.Part(text="a"),
                        usage=types.GenerateContentResponseUsageMetadata(
                            prompt_token_count=4, candidates_token_count=1
                        ),
                    ),
                    _chunk(
                        types.Part(text="b"),
                        usage=types.GenerateContentResponseUsageMetadata(
                            prompt_token_count=4,
                            candidates_token_count=2,
                            total_token_count=6,
                        ),
                    ),
                ]
            )
        )

        events = await _collect(model.stream([UserMessage(content="Hi")]))

        end = events[-1]
        assert end.usage is not None
        assert end.usage.completion_tokens == 2
        assert end.usage.total_tokens == 6

    @pytest.mark.asyncio
    async def test_retains_a_stop_reason_from_an_earlier_chunk(self) -> None:
        model = _model()
        model._client.models.generate_content_stream = AsyncMock(
            return_value=FakeChunkStream(
                [
                    _chunk(types.Part(text="a"), finish_reason="MAX_TOKENS"),
                    _chunk(types.Part(text="b")),
                ]
            )
        )

        events = await _collect(model.stream([UserMessage(content="Hi")]))

        assert events[-1].stop_reason == "MAX_TOKENS"

    @pytest.mark.asyncio
    async def test_still_emits_an_end_event_for_an_empty_stream(self) -> None:
        model = _model()
        model._client.models.generate_content_stream = AsyncMock(
            return_value=FakeChunkStream([])
        )

        events = await _collect(model.stream([UserMessage(content="Hi")]))

        assert len(events) == 1
        assert isinstance(events[0], StreamEnd)
        assert events[0].completion == ""
        assert events[0].usage is None
        assert events[0].tool_calls == []


class TestGoogleStreamRetries:
    @pytest.mark.asyncio
    async def test_retries_before_the_first_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(retries_provider.asyncio, "sleep", AsyncMock())
        callback = AsyncMock()
        model = _model(max_retries=1, on_retry=callback)
        model._client.models.generate_content_stream = AsyncMock(
            side_effect=[
                errors.APIError(503, {"message": "unavailable"}),
                FakeChunkStream([_chunk(types.Part(text="recovered"))]),
            ]
        )

        events = await _collect(model.stream([UserMessage(content="Hi")]))

        assert events[-1].completion == "recovered"
        assert model._client.models.generate_content_stream.await_count == 2
        callback.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_does_not_replay_a_stream_that_already_emitted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Retrying here would duplicate the text the caller already consumed.
        monkeypatch.setattr(retries_provider.asyncio, "sleep", AsyncMock())
        model = _model(max_retries=1)
        model._client.models.generate_content_stream = AsyncMock(
            return_value=FakeChunkStream(
                [_chunk(types.Part(text="partial"))],
                raise_after=errors.APIError(503, {"message": "unavailable"}),
            )
        )

        events = []
        with pytest.raises(RetryableError):
            async for event in model.stream([UserMessage(content="Hi")]):
                events.append(event)

        assert [event.delta for event in events] == ["partial"]
        assert model._client.models.generate_content_stream.await_count == 1
