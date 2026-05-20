import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("anthropic")

from llmify.base import ChatModel
from llmify.messages import UserMessage
from llmify.providers.anthropic import ChatAnthropic
from llmify.views import StreamEnd, StreamTextDelta, StreamToolCall


class FakeEventStream:
    def __init__(self, events: list[SimpleNamespace]):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    def __aiter__(self):
        async def generator():
            for event in self._events:
                yield event

        return generator()


class MockAnthropicModel(ChatAnthropic):
    def __init__(self):
        ChatModel.__init__(self)
        self._client = SimpleNamespace(messages=SimpleNamespace())
        self._model = "claude-test"


class TestAnthropicStreaming:
    @pytest.mark.asyncio
    async def test_streams_interleaved_text_tool_and_end(self) -> None:
        model = MockAnthropicModel()

        events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=12,
                        cache_read_input_tokens=2,
                        cache_creation_input_tokens=1,
                    )
                ),
            ),
            SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(type="text"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="Hi "),
            ),
            SimpleNamespace(
                type="content_block_start",
                index=1,
                content_block=SimpleNamespace(
                    type="tool_use", id="call_1", name="get_weather"
                ),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=1,
                delta=SimpleNamespace(
                    type="input_json_delta", partial_json='{"city":"Ber'
                ),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=1,
                delta=SimpleNamespace(type="input_json_delta", partial_json='lin"}'),
            ),
            SimpleNamespace(type="content_block_stop", index=1),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="there"),
            ),
            SimpleNamespace(type="content_block_stop", index=0),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="tool_use"),
                usage=SimpleNamespace(output_tokens=5),
            ),
            SimpleNamespace(type="message_stop"),
        ]

        model._client.messages.stream = Mock(return_value=FakeEventStream(events))

        observed = []
        async for event in model.stream([UserMessage(content="Hi")]):
            observed.append(event)

        assert [event.type for event in observed] == [
            "text",
            "tool_call",
            "text",
            "end",
        ]
        assert isinstance(observed[0], StreamTextDelta)
        assert isinstance(observed[1], StreamToolCall)
        assert isinstance(observed[3], StreamEnd)

        assert observed[0].delta == "Hi "
        assert observed[2].delta == "there"
        assert json.loads(observed[1].tool_call.function.arguments) == {
            "city": "Berlin"
        }

        end_event = observed[3]
        assert end_event.completion == "Hi there"
        assert end_event.stop_reason == "tool_use"
        assert end_event.usage is not None
        assert end_event.usage.prompt_tokens == 12
        assert end_event.usage.completion_tokens == 5
        assert end_event.usage.total_tokens == 17
        assert end_event.usage.prompt_cached_tokens == 2
        assert end_event.usage.prompt_cache_creation_tokens == 1

    @pytest.mark.asyncio
    async def test_defaults_empty_tool_json_to_empty_object(self) -> None:
        model = MockAnthropicModel()

        events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=1,
                        cache_read_input_tokens=None,
                        cache_creation_input_tokens=None,
                    )
                ),
            ),
            SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(
                    type="tool_use", id="call_2", name="ping"
                ),
            ),
            SimpleNamespace(type="content_block_stop", index=0),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="tool_use"),
                usage=SimpleNamespace(output_tokens=1),
            ),
        ]

        model._client.messages.stream = Mock(return_value=FakeEventStream(events))

        observed = []
        async for event in model.stream([UserMessage(content="Call ping")]):
            observed.append(event)

        assert isinstance(observed[0], StreamToolCall)
        assert observed[0].tool_call.function.arguments == "{}"

        end_event = observed[1]
        assert isinstance(end_event, StreamEnd)
        assert len(end_event.tool_calls) == 1
        assert end_event.tool_calls[0].function.arguments == "{}"

    @pytest.mark.asyncio
    async def test_passes_tool_choice_only_when_tools_present(self) -> None:
        model = MockAnthropicModel()

        model._client.messages.stream = Mock(return_value=FakeEventStream([]))
        async for _ in model.stream(
            [UserMessage(content="Hi")], tool_choice="required"
        ):
            pass

        first_call = model._client.messages.stream.call_args.kwargs
        assert "tools" not in first_call
        assert "tool_choice" not in first_call

        model._client.messages.stream = Mock(return_value=FakeEventStream([]))
        async for _ in model.stream(
            [UserMessage(content="Hi")],
            tools=[{"name": "x", "description": "", "input_schema": {}}],
            tool_choice="required",
        ):
            pass

        second_call = model._client.messages.stream.call_args.kwargs
        assert second_call["tool_choice"] == {"type": "any"}
        assert len(second_call["tools"]) == 1
