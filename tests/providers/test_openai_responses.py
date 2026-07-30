from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("openai")

import httpx
from openai import APIError
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)

import llmify.retries as retries_provider
from llmify import ChatOpenAIResponses, RetryEvent
from llmify.exceptions import RetryableError
from llmify.messages import (
    AssistantMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    Function,
    ImageURL,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from llmify.providers import openai_responses as responses_provider
from llmify.providers.openai_responses import _convert_messages, _convert_tools
from llmify.tools import FunctionTool
from llmify.views import StreamEnd, StreamTextDelta, StreamToolCall


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def _response(*, usage: ResponseUsage | None = None) -> Response:
    return Response.model_construct(
        status="completed",
        incomplete_details=None,
        error=None,
        usage=usage,
    )


def _text_delta(delta: str, sequence_number: int) -> ResponseTextDeltaEvent:
    return ResponseTextDeltaEvent.model_construct(
        content_index=0,
        delta=delta,
        item_id="item_1",
        logprobs=[],
        output_index=0,
        sequence_number=sequence_number,
        type="response.output_text.delta",
    )


def _completed(response: Response, sequence_number: int) -> ResponseCompletedEvent:
    return ResponseCompletedEvent.model_construct(
        response=response,
        sequence_number=sequence_number,
        type="response.completed",
    )


async def _stream(*events):
    for event in events:
        yield event


async def _failing_stream(*events):
    for event in events:
        yield event
    raise _overloaded_error()


def _overloaded_error() -> APIError:
    return APIError(
        "Our servers are currently overloaded. Please try again later.",
        httpx.Request("POST", "https://api.openai.com/v1/responses"),
        body={"type": "server_error"},
    )


class TestMessageConversion:
    def test_splits_instructions_and_input_items(self) -> None:
        messages = [
            SystemMessage(content="First"),
            SystemMessage(content=[ContentPartTextParam(text="Second")]),
            UserMessage(
                content=[
                    ContentPartTextParam(text="What is this?"),
                    ContentPartImageParam(
                        image_url=ImageURL(url="https://example.com/image.png")
                    ),
                ]
            ),
            AssistantMessage(
                content="Let me check.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function=Function(name="inspect", arguments='{"id": 1}'),
                    )
                ],
            ),
            ToolResultMessage(tool_call_id="call_1", content="done"),
        ]

        instructions, items = _convert_messages(messages)

        assert instructions == "First\n\nSecond"
        assert items == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is this?"},
                    {
                        "type": "input_image",
                        "image_url": "https://example.com/image.png",
                        "detail": "auto",
                    },
                ],
            },
            {"role": "assistant", "content": "Let me check."},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "inspect",
                "arguments": '{"id": 1}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "done",
            },
        ]

    def test_flattens_shared_tool_schema(self) -> None:
        def lookup(query: str) -> str:
            """Look something up."""
            return query

        converted = _convert_tools([FunctionTool(lookup)])

        assert converted[0]["type"] == "function"
        assert converted[0]["name"] == "lookup"
        assert converted[0]["description"] == "Look something up."
        assert converted[0]["strict"] is False


class TestConfiguration:
    def test_rejects_stream_as_constructor_parameter(self) -> None:
        with pytest.raises(TypeError, match="'stream' is managed"):
            ChatOpenAIResponses(model="gpt-test", stream=False)

    @pytest.mark.asyncio
    async def test_rejects_stream_as_method_parameter(self) -> None:
        model = ChatOpenAIResponses(model="gpt-test")

        with pytest.raises(TypeError, match="'stream' is managed"):
            await model.invoke([UserMessage(content="Hi")], stream=False)

    @pytest.mark.parametrize("max_retries", [-1, -5])
    def test_rejects_negative_max_retries(self, max_retries: int) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            ChatOpenAIResponses(model="gpt-test", max_retries=max_retries)

    @pytest.mark.parametrize("max_retries", [True, 1.5, "2"])
    def test_rejects_non_integer_max_retries(self, max_retries) -> None:
        with pytest.raises(TypeError, match="must be an integer"):
            ChatOpenAIResponses(model="gpt-test", max_retries=max_retries)

    def test_llmify_owns_the_responses_retry_budget(self) -> None:
        model = ChatOpenAIResponses(model="gpt-test", max_retries=4)

        assert model._default_max_retries == 4
        assert model._client.max_retries == 0


class TestInvoke:
    @pytest.mark.asyncio
    async def test_collects_stream_and_translates_request_parameters(self) -> None:
        usage = ResponseUsage.model_construct(
            input_tokens=5,
            input_tokens_details=SimpleNamespace(cached_tokens=2),
            output_tokens=3,
            total_tokens=8,
        )
        response = _response(usage=usage)
        events = _stream(
            _text_delta("Hello", 0),
            _text_delta(" world", 1),
            _completed(response, 2),
        )
        model = ChatOpenAIResponses(
            model="gpt-test",
            max_tokens=20,
            frequency_penalty=0.5,
        )
        model._client.responses.create = AsyncMock(return_value=events)

        result = await model.invoke([UserMessage(content="Hi")])

        assert result.completion == "Hello world"
        assert result.stop_reason == "completed"
        assert result.usage is not None
        assert result.usage.prompt_cached_tokens == 2

        request = model._client.responses.create.call_args.kwargs
        assert request["stream"] is True
        assert request["max_output_tokens"] == 20
        assert "max_tokens" not in request
        assert "frequency_penalty" not in request

    @pytest.mark.asyncio
    async def test_retries_and_discards_an_incomplete_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeper = AsyncMock()
        monkeypatch.setattr(responses_provider, "sleep_before_retry", sleeper)
        model = ChatOpenAIResponses(model="gpt-test", max_retries=2)
        model._client.responses.create = AsyncMock(
            side_effect=[
                _failing_stream(_text_delta("discarded", 0)),
                _stream(
                    _text_delta("Hello", 0),
                    _completed(_response(), 1),
                ),
            ]
        )

        result = await model.invoke([UserMessage(content="Hi")])

        assert result.completion == "Hello"
        assert model._client.responses.create.await_count == 2
        sleeper.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_does_not_retry_when_disabled(self) -> None:
        model = ChatOpenAIResponses(model="gpt-test", max_retries=0)
        model._client.responses.create = AsyncMock(return_value=_failing_stream())

        with pytest.raises(RetryableError, match="overloaded"):
            await model.invoke([UserMessage(content="Hi")])

        assert model._client.responses.create.await_count == 1

    @pytest.mark.asyncio
    async def test_reports_request_retries_to_sync_callback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[RetryEvent] = []
        sleep = AsyncMock()
        monkeypatch.setattr(retries_provider.asyncio, "sleep", sleep)
        monkeypatch.setattr(retries_provider.random, "uniform", lambda _a, _b: 1.0)
        model = ChatOpenAIResponses(
            model="gpt-test",
            max_retries=3,
            on_retry=events.append,
        )
        model._client.responses.create = AsyncMock(
            side_effect=[
                _overloaded_error(),
                _stream(_completed(_response(), 0)),
            ]
        )

        await model.invoke([UserMessage(content="Hi")])

        assert len(events) == 1
        event = events[0]
        assert event.retry_number == 1
        assert event.max_retries == 3
        assert event.delay == 0.5
        assert event.failed_attempt == 1
        assert event.next_attempt == 2
        assert event.max_attempts == 4
        assert "overloaded" in str(event.error)
        sleep.assert_awaited_once_with(0.5)

    @pytest.mark.asyncio
    async def test_per_call_async_callback_overrides_client_callback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        default_callback = AsyncMock()
        call_callback = AsyncMock()
        monkeypatch.setattr(retries_provider.asyncio, "sleep", AsyncMock())
        model = ChatOpenAIResponses(
            model="gpt-test",
            max_retries=1,
            on_retry=default_callback,
        )
        model._client.responses.create = AsyncMock(
            side_effect=[
                _overloaded_error(),
                _stream(_completed(_response(), 0)),
            ]
        )

        await model.invoke(
            [UserMessage(content="Hi")],
            on_retry=call_callback,
        )

        call_callback.assert_awaited_once()
        default_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_callback_can_cancel_the_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleep = AsyncMock()
        monkeypatch.setattr(retries_provider.asyncio, "sleep", sleep)

        def cancel(_event: RetryEvent) -> None:
            raise RuntimeError("cancel retry")

        model = ChatOpenAIResponses(
            model="gpt-test",
            max_retries=2,
            on_retry=cancel,
        )
        model._client.responses.create = AsyncMock(side_effect=_overloaded_error())

        with pytest.raises(RuntimeError, match="cancel retry"):
            await model.invoke([UserMessage(content="Hi")])

        assert model._client.responses.create.await_count == 1
        sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sends_the_reasoning_effort(self) -> None:
        model = ChatOpenAIResponses(model="gpt-test", reasoning_effort="high")
        model._client.responses.create = AsyncMock(
            return_value=_stream(_completed(_response(), 0))
        )

        await model.invoke([UserMessage(content="Hi")])

        request = model._client.responses.create.call_args.kwargs
        assert request["reasoning"] == {"effort": "high"}
        assert "reasoning_effort" not in request

    @pytest.mark.asyncio
    async def test_per_call_reasoning_effort_wins(self) -> None:
        model = ChatOpenAIResponses(model="gpt-test", reasoning_effort="high")
        model._client.responses.create = AsyncMock(
            return_value=_stream(_completed(_response(), 0))
        )

        await model.invoke([UserMessage(content="Hi")], reasoning_effort="low")

        request = model._client.responses.create.call_args.kwargs
        assert request["reasoning"] == {"effort": "low"}

    @pytest.mark.asyncio
    async def test_omits_reasoning_when_not_configured(self) -> None:
        model = ChatOpenAIResponses(model="gpt-test")
        model._client.responses.create = AsyncMock(
            return_value=_stream(_completed(_response(), 0))
        )

        await model.invoke([UserMessage(content="Hi")])

        assert "reasoning" not in model._client.responses.create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_emits_complete_tool_call(self) -> None:
        item = ResponseFunctionToolCall.model_construct(
            type="function_call",
            call_id="call_1",
            id="item_1",
            name="lookup",
            arguments='{"query": "test"}',
        )
        response = _response()
        output_done = ResponseOutputItemDoneEvent.model_construct(
            item=item,
            output_index=0,
            sequence_number=0,
            type="response.output_item.done",
        )
        events = _stream(
            output_done,
            _completed(response, 1),
        )
        model = ChatOpenAIResponses(model="gpt-test")
        model._client.responses.create = AsyncMock(return_value=events)

        emitted = []
        async for event in model.stream([UserMessage(content="Hi")]):
            emitted.append(event)

        assert isinstance(emitted[0], StreamToolCall)
        assert emitted[0].tool_call.id == "call_1"
        assert emitted[0].tool_call.function.name == "lookup"
        assert isinstance(emitted[-1], StreamEnd)
        assert emitted[-1].tool_calls == [emitted[0].tool_call]

    @pytest.mark.asyncio
    async def test_emits_text_delta_and_end(self) -> None:
        response = _response()
        events = _stream(
            _text_delta("Hi", 0),
            _completed(response, 1),
        )
        model = ChatOpenAIResponses(model="gpt-test")
        model._client.responses.create = AsyncMock(return_value=events)

        emitted = []
        async for event in model.stream([UserMessage(content="Hi")]):
            emitted.append(event)

        assert isinstance(emitted[0], StreamTextDelta)
        assert emitted[0].delta == "Hi"
        assert isinstance(emitted[1], StreamEnd)
        assert emitted[1].completion == "Hi"

    @pytest.mark.asyncio
    async def test_stream_retries_before_emitting_an_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sleeper = AsyncMock()
        monkeypatch.setattr(responses_provider, "sleep_before_retry", sleeper)
        model = ChatOpenAIResponses(model="gpt-test", max_retries=2)
        model._client.responses.create = AsyncMock(
            side_effect=[
                _failing_stream(),
                _stream(
                    _text_delta("Hello", 0),
                    _completed(_response(), 1),
                ),
            ]
        )

        emitted = [event async for event in model.stream([UserMessage(content="Hi")])]

        assert emitted[0] == StreamTextDelta(delta="Hello")
        assert model._client.responses.create.await_count == 2
        sleeper.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stream_does_not_replay_after_emitting_an_event(self) -> None:
        model = ChatOpenAIResponses(model="gpt-test", max_retries=2)
        model._client.responses.create = AsyncMock(
            return_value=_failing_stream(_text_delta("partial", 0))
        )
        emitted = []

        with pytest.raises(RetryableError, match="overloaded"):
            async for event in model.stream([UserMessage(content="Hi")]):
                emitted.append(event)

        assert emitted == [StreamTextDelta(delta="partial")]
        assert model._client.responses.create.await_count == 1
