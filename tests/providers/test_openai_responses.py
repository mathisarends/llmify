from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("openai")

from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)

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
from llmify.providers.openai_responses import (
    OpenAIResponses,
    _convert_messages,
    _convert_tools,
)
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
            OpenAIResponses(model="gpt-test", stream=False)

    @pytest.mark.asyncio
    async def test_rejects_stream_as_method_parameter(self) -> None:
        model = OpenAIResponses(model="gpt-test")

        with pytest.raises(TypeError, match="'stream' is managed"):
            await model.invoke([UserMessage(content="Hi")], stream=False)


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
        model = OpenAIResponses(
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
        model = OpenAIResponses(model="gpt-test")
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
        model = OpenAIResponses(model="gpt-test")
        model._client.responses.create = AsyncMock(return_value=events)

        emitted = []
        async for event in model.stream([UserMessage(content="Hi")]):
            emitted.append(event)

        assert isinstance(emitted[0], StreamTextDelta)
        assert emitted[0].delta == "Hi"
        assert isinstance(emitted[1], StreamEnd)
        assert emitted[1].completion == "Hi"
