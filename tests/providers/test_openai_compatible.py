import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

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
from llmify.providers import ChatInvokeCompletion
from llmify.providers.openai_compatible import OpenAICompatible
from llmify.tools import FunctionTool
from llmify.views import StreamEnd, StreamTextDelta, StreamToolCall


class SearchResult(BaseModel):
    query: str
    results: list[str]


class MockChatModel(OpenAICompatible):
    def __init__(self, **kwargs):
        kwargs.setdefault("model", "gpt-4")
        super().__init__(**kwargs)
        self._client = AsyncMock()


@pytest.fixture
def mock_model() -> MockChatModel:
    return MockChatModel()


@pytest.fixture
def search_tool() -> FunctionTool:
    def search_web(query: str) -> str:
        """Search the web"""
        return f"Results for: {query}"

    return FunctionTool(search_web)


class TestMessageConversion:
    def test_converts_user_message(self, mock_model: MockChatModel) -> None:
        messages = [UserMessage(content="Hello")]
        converted = mock_model._convert_messages(messages)

        assert converted == [{"role": "user", "content": "Hello"}]

    def test_converts_system_message(self, mock_model: MockChatModel) -> None:
        messages = [SystemMessage(content="You are helpful")]
        converted = mock_model._convert_messages(messages)

        assert converted == [{"role": "system", "content": "You are helpful"}]

    def test_converts_assistant_message(self, mock_model: MockChatModel) -> None:
        messages = [AssistantMessage(content="I can help")]
        converted = mock_model._convert_messages(messages)

        assert converted == [{"role": "assistant", "content": "I can help"}]

    def test_converts_user_message_with_image(self, mock_model: MockChatModel) -> None:
        message = UserMessage(
            content=[
                ContentPartTextParam(text="What's this?"),
                ContentPartImageParam(
                    image_url=ImageURL(
                        url="data:image/png;base64,iVBORw0KG...", media_type="image/png"
                    )
                ),
            ]
        )
        converted = mock_model._convert_single_message(message)

        assert converted["role"] == "user"
        assert len(converted["content"]) == 2
        assert converted["content"][0]["type"] == "text"
        assert converted["content"][1]["type"] == "image_url"
        assert "data:image/png;base64," in converted["content"][1]["image_url"]["url"]

    def test_converts_tool_result_message(self, mock_model: MockChatModel) -> None:
        message = ToolResultMessage(tool_call_id="call_123", content="Search completed")
        converted = mock_model._convert_single_message(message)

        assert converted == {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "Search completed",
        }

    def test_converts_assistant_message_with_tool_calls(
        self, mock_model: MockChatModel
    ) -> None:
        message = AssistantMessage(
            content="Let me search",
            tool_calls=[
                ToolCall(
                    id="call_123",
                    function=Function(
                        name="search", arguments=json.dumps({"query": "test"})
                    ),
                )
            ],
        )
        converted = mock_model._convert_single_message(message)

        assert converted["role"] == "assistant"
        assert converted["content"] == "Let me search"
        assert len(converted["tool_calls"]) == 1
        assert converted["tool_calls"][0]["id"] == "call_123"
        assert converted["tool_calls"][0]["function"]["name"] == "search"

    def test_converts_assistant_message_with_serialized_pydantic_arguments(
        self, mock_model: MockChatModel
    ) -> None:
        class SearchParams(BaseModel):
            query: str
            max_results: int

        params = SearchParams(query="test", max_results=10)
        message = AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_456",
                    function=Function(
                        name="search", arguments=params.model_dump_json()
                    ),
                )
            ],
        )
        converted = mock_model._convert_single_message(message)

        parsed_args = json.loads(converted["tool_calls"][0]["function"]["arguments"])
        assert converted["tool_calls"][0]["function"]["name"] == "search"
        assert parsed_args["query"] == "test"
        assert parsed_args["max_results"] == 10


class TestParameterMerging:
    def test_merges_default_and_method_params(self) -> None:
        model = MockChatModel(temperature=0.5, max_tokens=100)
        params = model._merge_params({"top_p": 0.9})

        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 100
        assert params["top_p"] == 0.9

    def test_method_params_override_defaults(self) -> None:
        model = MockChatModel(temperature=0.5)
        params = model._merge_params({"temperature": 0.8})

        assert params["temperature"] == 0.8

    def test_excludes_none_values(self) -> None:
        model = MockChatModel(temperature=0.5)
        params = model._merge_params({"top_p": 0.9})

        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9

    def test_preserves_custom_kwargs(self) -> None:
        model = MockChatModel(custom_param="value")
        params = model._merge_params({})

        assert params["custom_param"] == "value"


class TestPlainInvoke:
    @pytest.mark.asyncio
    async def test_returns_completion_with_text_content(
        self, mock_model: MockChatModel
    ) -> None:
        mock_response = Mock(spec=ChatCompletion)
        mock_response.choices = [
            Mock(message=Mock(content="Hello world"), finish_reason="stop")
        ]
        mock_response.usage = None
        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await mock_model.invoke([UserMessage(content="Hi")])

        assert result.completion == "Hello world"

    @pytest.mark.asyncio
    async def test_returns_empty_string_for_none_content(
        self, mock_model: MockChatModel
    ) -> None:
        mock_response = Mock(spec=ChatCompletion)
        mock_response.choices = [Mock(message=Mock(content=None), finish_reason="stop")]
        mock_response.usage = None
        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await mock_model.invoke([UserMessage(content="Hi")])

        assert result.completion == ""

    @pytest.mark.asyncio
    async def test_passes_merged_parameters(self, mock_model: MockChatModel) -> None:
        mock_response = Mock(spec=ChatCompletion)
        mock_response.choices = [
            Mock(message=Mock(content="Response"), finish_reason="stop")
        ]
        mock_response.usage = None
        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        await mock_model.invoke(
            [UserMessage(content="Hi")], temperature=0.7, max_tokens=50
        )

        call_kwargs = mock_model._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 50


class TestToolInvoke:
    @pytest.mark.asyncio
    async def test_returns_completion_with_tool_calls(
        self, mock_model: MockChatModel, search_tool: FunctionTool
    ) -> None:
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "search_web"
        mock_tool_call.function.arguments = '{"query": "test"}'

        mock_response = Mock(spec=ChatCompletion)
        mock_response.choices = [
            Mock(
                message=Mock(content="Searching...", tool_calls=[mock_tool_call]),
                finish_reason="tool_calls",
            )
        ]
        mock_response.usage = None
        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await mock_model.invoke(
            [UserMessage(content="Search for test")], tools=[search_tool]
        )

        assert isinstance(result, ChatInvokeCompletion)
        assert result.completion == "Searching..."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "search_web"
        assert json.loads(result.tool_calls[0].function.arguments) == {"query": "test"}

    @pytest.mark.asyncio
    async def test_returns_empty_tool_calls_when_none(
        self, mock_model: MockChatModel, search_tool: FunctionTool
    ) -> None:
        mock_response = Mock(spec=ChatCompletion)
        mock_response.choices = [
            Mock(message=Mock(content="Done", tool_calls=None), finish_reason="stop")
        ]
        mock_response.usage = None
        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await mock_model.invoke(
            [UserMessage(content="Hi")], tools=[search_tool]
        )

        assert isinstance(result, ChatInvokeCompletion)
        assert result.completion == "Done"
        assert result.tool_calls == []


class TestStructuredOutput:
    @pytest.mark.asyncio
    async def test_returns_parsed_pydantic_model(
        self, mock_model: MockChatModel
    ) -> None:
        expected_result = SearchResult(query="test", results=["a", "b"])
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(parsed=expected_result), finish_reason="stop")
        ]
        mock_response.usage = None
        mock_model._client.beta.chat.completions.parse = AsyncMock(
            return_value=mock_response
        )

        result = await mock_model.invoke(
            [UserMessage(content="Search")], output_format=SearchResult
        )

        assert isinstance(result.completion, SearchResult)
        assert result.completion.query == "test"
        assert result.completion.results == ["a", "b"]

    @pytest.mark.asyncio
    async def test_uses_beta_parse_endpoint_with_response_format(
        self, mock_model: MockChatModel
    ) -> None:
        expected_result = SearchResult(query="test", results=[])
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(parsed=expected_result), finish_reason="stop")
        ]
        mock_response.usage = None
        mock_model._client.beta.chat.completions.parse = AsyncMock(
            return_value=mock_response
        )

        await mock_model.invoke(
            [UserMessage(content="Search")], output_format=SearchResult
        )

        mock_model._client.beta.chat.completions.parse.assert_called_once()
        call_kwargs = mock_model._client.beta.chat.completions.parse.call_args.kwargs
        assert call_kwargs["response_format"] is SearchResult


class TestStreaming:
    @staticmethod
    def _make_chunk(
        *,
        content: str | None = None,
        tool_calls: list | None = None,
        finish_reason: str | None = None,
        usage: Mock | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=content, tool_calls=tool_calls),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )

    @pytest.mark.asyncio
    async def test_streams_text_and_end_event(self, mock_model: MockChatModel) -> None:
        usage = Mock(
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
            prompt_tokens_details=Mock(cached_tokens=3),
        )

        async def mock_stream():
            yield self._make_chunk(content="Hello")
            yield self._make_chunk(content=" world")
            yield self._make_chunk(finish_reason="stop", usage=usage)

        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )

        events = []
        async for event in mock_model.stream([UserMessage(content="Hi")]):
            events.append(event)

        assert isinstance(events[0], StreamTextDelta)
        assert isinstance(events[1], StreamTextDelta)
        assert isinstance(events[2], StreamEnd)
        assert events[0].delta == "Hello"
        assert events[1].delta == " world"
        assert events[2].completion == "Hello world"
        assert events[2].usage is not None
        assert events[2].usage.total_tokens == 14

    @pytest.mark.asyncio
    async def test_buffers_tool_calls_until_finish(
        self, mock_model: MockChatModel
    ) -> None:
        async def mock_stream():
            yield self._make_chunk(
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id="call_1",
                        function=SimpleNamespace(
                            name="search_web", arguments='{"query":"te'
                        ),
                    )
                ]
            )
            yield self._make_chunk(
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id=None,
                        function=SimpleNamespace(name=None, arguments='st"}'),
                    )
                ]
            )
            yield self._make_chunk(finish_reason="tool_calls")

        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )

        events = []
        async for event in mock_model.stream([UserMessage(content="Hi")]):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], StreamToolCall)
        assert isinstance(events[1], StreamEnd)
        assert json.loads(events[0].tool_call.function.arguments) == {"query": "test"}
        assert len(events[1].tool_calls) == 1
        assert events[1].tool_calls[0].function.name == "search_web"

    @pytest.mark.asyncio
    async def test_handles_multiple_tool_calls_in_stable_order(
        self, mock_model: MockChatModel
    ) -> None:
        async def mock_stream():
            yield self._make_chunk(content="Working...")
            yield self._make_chunk(
                tool_calls=[
                    SimpleNamespace(
                        index=1,
                        id="call_b",
                        function=SimpleNamespace(name="tool_b", arguments='{"b":'),
                    ),
                    SimpleNamespace(
                        index=0,
                        id="call_a",
                        function=SimpleNamespace(name="tool_a", arguments='{"a":1}'),
                    ),
                ]
            )
            yield self._make_chunk(
                tool_calls=[
                    SimpleNamespace(
                        index=1,
                        id=None,
                        function=SimpleNamespace(name=None, arguments="2}"),
                    )
                ]
            )
            yield self._make_chunk(finish_reason="tool_calls")

        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )

        events = []
        async for event in mock_model.stream([UserMessage(content="Hi")]):
            events.append(event)

        assert [event.type for event in events] == [
            "text",
            "tool_call",
            "tool_call",
            "end",
        ]
        assert events[1].tool_call.id == "call_a"
        assert events[2].tool_call.id == "call_b"
        assert json.loads(events[1].tool_call.function.arguments) == {"a": 1}
        assert json.loads(events[2].tool_call.function.arguments) == {"b": 2}

        end_event = events[3]
        assert isinstance(end_event, StreamEnd)
        assert [tc.id for tc in end_event.tool_calls] == ["call_a", "call_b"]
        assert end_event.completion == "Working..."

    @pytest.mark.asyncio
    async def test_ignores_tool_choice_when_tools_are_missing(
        self, mock_model: MockChatModel
    ) -> None:
        async def mock_stream():
            yield self._make_chunk(finish_reason="stop")

        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )

        async for _ in mock_model.stream(
            [UserMessage(content="Hi")], tool_choice="required"
        ):
            pass

        call_kwargs = mock_model._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["stream_options"]["include_usage"] is True
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs

    @pytest.mark.asyncio
    async def test_passes_tools_and_tool_choice_when_tools_present(
        self, mock_model: MockChatModel, search_tool: FunctionTool
    ) -> None:
        async def mock_stream():
            yield self._make_chunk(finish_reason="stop")

        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )

        async for _ in mock_model.stream(
            [UserMessage(content="Hi")],
            tools=[search_tool],
            tool_choice="required",
        ):
            pass

        call_kwargs = mock_model._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == "required"
        assert len(call_kwargs["tools"]) == 1

    @pytest.mark.asyncio
    async def test_tolerates_usage_only_chunk(self, mock_model: MockChatModel) -> None:
        usage = Mock(
            prompt_tokens=5,
            completion_tokens=2,
            total_tokens=7,
            prompt_tokens_details=Mock(cached_tokens=0),
        )

        async def mock_stream():
            yield SimpleNamespace(choices=[], usage=usage)
            yield self._make_chunk(content="done", finish_reason="stop")

        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )

        events = []
        async for event in mock_model.stream([UserMessage(content="Hi")]):
            events.append(event)

        end_event = events[-1]
        assert isinstance(end_event, StreamEnd)
        assert end_event.usage is not None
        assert end_event.usage.total_tokens == 7

    @pytest.mark.asyncio
    async def test_passes_stream_parameter(self, mock_model: MockChatModel) -> None:
        async def mock_stream():
            yield self._make_chunk(finish_reason="stop")

        mock_model._client.chat.completions.create = AsyncMock(
            return_value=mock_stream()
        )

        async for _ in mock_model.stream([UserMessage(content="Hi")]):
            pass

        call_kwargs = mock_model._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True
