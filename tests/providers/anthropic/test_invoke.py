import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

pytest.importorskip("anthropic")

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
from llmify.providers.anthropic import ChatAnthropic
from llmify.providers.anthropic.client import (
    _build_params,
    _convert_messages,
    _convert_tool,
)
from llmify.tools import tool


@tool
def get_weather(city: str) -> str:
    """Look up the weather"""
    return f"sunny in {city}"


def _model(**kwargs: Any) -> ChatAnthropic:
    client = SimpleNamespace(messages=SimpleNamespace())
    kwargs.setdefault("max_retries", 0)
    return ChatAnthropic(model="claude-test", client=client, **kwargs)


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(block_id: str, name: str, payload: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=payload)


def _response(
    *content: SimpleNamespace,
    stop_reason: str = "end_turn",
    input_tokens: int = 11,
    output_tokens: int = 7,
    cache_read: int | None = None,
    cache_creation: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=list(content),
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
        ),
    )


class TestInvokePlain:
    @pytest.mark.asyncio
    async def test_joins_all_text_blocks(self) -> None:
        model = _model()
        model._client.messages.create = AsyncMock(
            return_value=_response(_text_block("Hello "), _text_block("world"))
        )

        result = await model.invoke([UserMessage(content="Hi")])

        assert result.completion == "Hello world"
        assert result.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_reports_usage_including_cache_counters(self) -> None:
        model = _model()
        model._client.messages.create = AsyncMock(
            return_value=_response(
                _text_block("hi"),
                input_tokens=11,
                output_tokens=7,
                cache_read=3,
                cache_creation=2,
            )
        )

        usage = (await model.invoke([UserMessage(content="Hi")])).usage

        assert usage is not None
        assert usage.prompt_tokens == 11
        assert usage.completion_tokens == 7
        assert usage.total_tokens == 18
        assert usage.prompt_cached_tokens == 3
        assert usage.prompt_cache_creation_tokens == 2

    @pytest.mark.asyncio
    async def test_sends_no_tool_parameters(self) -> None:
        model = _model()
        model._client.messages.create = AsyncMock(return_value=_response())

        await model.invoke([UserMessage(content="Hi")])

        request = model._client.messages.create.call_args.kwargs
        assert "tools" not in request
        assert "tool_choice" not in request


class TestInvokeWithTools:
    @pytest.mark.asyncio
    async def test_parses_tool_use_blocks_into_tool_calls(self) -> None:
        model = _model()
        model._client.messages.create = AsyncMock(
            return_value=_response(
                _text_block("Checking. "),
                _tool_use_block("call_1", "get_weather", {"city": "Berlin"}),
                stop_reason="tool_use",
            )
        )

        result = await model.invoke(
            [UserMessage(content="Weather?")], tools=[get_weather]
        )

        assert result.completion == "Checking. "
        assert result.stop_reason == "tool_use"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_1"
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Berlin"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_choice", "expected"),
        [
            ("auto", {"type": "auto"}),
            ("required", {"type": "any"}),
            ("none", {"type": "none"}),
        ],
    )
    async def test_maps_tool_choice(self, tool_choice: str, expected: dict) -> None:
        model = _model()
        model._client.messages.create = AsyncMock(return_value=_response())

        await model.invoke(
            [UserMessage(content="Hi")], tools=[get_weather], tool_choice=tool_choice
        )

        request = model._client.messages.create.call_args.kwargs
        assert request["tool_choice"] == expected

    @pytest.mark.asyncio
    async def test_converts_tools_to_the_anthropic_schema(self) -> None:
        model = _model()
        model._client.messages.create = AsyncMock(return_value=_response())

        await model.invoke([UserMessage(content="Hi")], tools=[get_weather])

        sent = model._client.messages.create.call_args.kwargs["tools"]
        assert sent[0]["name"] == "get_weather"
        assert sent[0]["description"] == "Look up the weather"
        assert sent[0]["input_schema"]["properties"]["city"] == {"type": "string"}

    @pytest.mark.asyncio
    async def test_passes_raw_dict_tools_through(self) -> None:
        model = _model()
        model._client.messages.create = AsyncMock(return_value=_response())
        raw = {"name": "web_search", "type": "web_search_20250305"}

        await model.invoke([UserMessage(content="Hi")], tools=[raw])

        assert model._client.messages.create.call_args.kwargs["tools"] == [raw]


class Answer(BaseModel):
    city: str
    degrees: int


class TestInvokeWithStructuredOutput:
    @pytest.mark.asyncio
    async def test_parses_the_structured_output_tool_call(self) -> None:
        model = _model()
        model._client.messages.create = AsyncMock(
            return_value=_response(
                _tool_use_block(
                    "call_1", "structured_output", {"city": "Berlin", "degrees": 18}
                )
            )
        )

        result = await model.invoke([UserMessage(content="Weather?")], Answer)

        assert result.completion == Answer(city="Berlin", degrees=18)

    @pytest.mark.asyncio
    async def test_forces_the_structured_output_tool(self) -> None:
        model = _model()
        model._client.messages.create = AsyncMock(
            return_value=_response(
                _tool_use_block(
                    "call_1", "structured_output", {"city": "B", "degrees": 1}
                )
            )
        )

        await model.invoke([UserMessage(content="Weather?")], Answer)

        request = model._client.messages.create.call_args.kwargs
        assert request["tool_choice"] == {"type": "tool", "name": "structured_output"}
        assert request["tools"][0]["name"] == "structured_output"
        assert request["tools"][0]["input_schema"] == Answer.model_json_schema()

    @pytest.mark.asyncio
    async def test_skips_unrelated_tool_use_blocks(self) -> None:
        model = _model()
        model._client.messages.create = AsyncMock(
            return_value=_response(
                _tool_use_block("call_1", "get_weather", {"city": "Berlin"}),
                _tool_use_block(
                    "call_2", "structured_output", {"city": "Berlin", "degrees": 18}
                ),
            )
        )

        result = await model.invoke([UserMessage(content="Weather?")], Answer)

        assert result.completion.city == "Berlin"

    @pytest.mark.asyncio
    async def test_raises_when_the_model_answers_with_text(self) -> None:
        model = _model()
        model._client.messages.create = AsyncMock(
            return_value=_response(_text_block("I would rather not."))
        )

        with pytest.raises(ValueError, match="No structured output"):
            await model.invoke([UserMessage(content="Weather?")], Answer)

    @pytest.mark.asyncio
    async def test_takes_precedence_over_tools(self) -> None:
        model = _model()
        model._client.messages.create = AsyncMock(
            return_value=_response(
                _tool_use_block(
                    "call_1", "structured_output", {"city": "B", "degrees": 1}
                )
            )
        )

        await model.invoke([UserMessage(content="Hi")], Answer, tools=[get_weather])

        request = model._client.messages.create.call_args.kwargs
        assert [sent["name"] for sent in request["tools"]] == ["structured_output"]


class TestBuildParams:
    def test_defaults_max_tokens(self) -> None:
        params = _build_params("claude-test", [UserMessage(content="Hi")], {})
        assert params["max_tokens"] == 4096

    def test_replaces_a_falsy_max_tokens_with_the_default(self) -> None:
        params = _build_params("claude-test", [], {"max_tokens": 0})
        assert params["max_tokens"] == 4096

    def test_lifts_the_system_message_into_its_own_field(self) -> None:
        params = _build_params(
            "claude-test",
            [SystemMessage(content="You are terse."), UserMessage(content="Hi")],
            {},
        )

        assert params["system"] == "You are terse."
        assert params["messages"] == [{"role": "user", "content": "Hi"}]

    def test_omits_the_system_field_when_absent(self) -> None:
        params = _build_params("claude-test", [UserMessage(content="Hi")], {})
        assert "system" not in params

    def test_wraps_a_single_stop_string(self) -> None:
        params = _build_params("claude-test", [], {"stop": "END"})
        assert params["stop_sequences"] == ["END"]

    def test_keeps_a_stop_list(self) -> None:
        params = _build_params("claude-test", [], {"stop": ["A", "B"]})
        assert params["stop_sequences"] == ["A", "B"]

    def test_forwards_shared_sampling_params(self) -> None:
        params = _build_params("claude-test", [], {"temperature": 0.2, "top_p": 0.9})
        assert params["temperature"] == 0.2
        assert params["top_p"] == 0.9

    @pytest.mark.parametrize(
        "unsupported",
        ["frequency_penalty", "presence_penalty", "seed", "response_format"],
    )
    def test_drops_params_anthropic_does_not_accept(self, unsupported: str) -> None:
        params = _build_params("claude-test", [], {unsupported: 1})
        assert unsupported not in params

    def test_forwards_unknown_params_untouched(self) -> None:
        params = _build_params("claude-test", [], {"thinking": {"type": "enabled"}})
        assert params["thinking"] == {"type": "enabled"}


class TestConvertMessages:
    def test_round_trips_a_tool_call_and_its_result(self) -> None:
        _, converted = _convert_messages(
            [
                AssistantMessage(
                    content="Checking.",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            function=Function(
                                name="get_weather", arguments='{"city": "Berlin"}'
                            ),
                        )
                    ],
                ),
                ToolResultMessage(tool_call_id="call_1", content="18 degrees"),
            ]
        )

        assert converted == [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Checking."},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "get_weather",
                        "input": {"city": "Berlin"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "18 degrees",
                    }
                ],
            },
        ]

    def test_omits_empty_assistant_text_before_a_tool_call(self) -> None:
        _, converted = _convert_messages(
            [
                AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            function=Function(name="ping", arguments="{}"),
                        )
                    ]
                )
            ]
        )

        assert [part["type"] for part in converted[0]["content"]] == ["tool_use"]

    def test_inlines_base64_images(self) -> None:
        _, converted = _convert_messages(
            [
                UserMessage(
                    content=[
                        ContentPartTextParam(text="What is this?"),
                        ContentPartImageParam(
                            image_url=ImageURL(url="data:image/png;base64,abc123")
                        ),
                    ]
                )
            ]
        )

        assert converted[0]["content"] == [
            {"type": "text", "text": "What is this?"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "abc123",
                },
            },
        ]

    def test_references_remote_images_by_url(self) -> None:
        _, converted = _convert_messages(
            [
                UserMessage(
                    content=[
                        ContentPartImageParam(
                            image_url=ImageURL(url="https://example.com/cat.png")
                        )
                    ]
                )
            ]
        )

        assert converted[0]["content"] == [
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/cat.png"},
            }
        ]

    def test_flattens_plain_string_messages(self) -> None:
        _, converted = _convert_messages(
            [UserMessage(content="Hi"), AssistantMessage(content="Hello")]
        )

        assert converted == [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]


class TestConvertTool:
    def test_translates_an_openai_function_schema(self) -> None:
        converted = _convert_tool(get_weather)

        assert converted["name"] == "get_weather"
        assert converted["description"] == "Look up the weather"
        assert converted["input_schema"]["required"] == ["city"]


class TestClientInjection:
    def test_uses_the_injected_client(self) -> None:
        client = SimpleNamespace(messages=SimpleNamespace())
        model = ChatAnthropic(model="claude-test", client=client)
        assert model._client is client

    def test_builds_its_own_client_by_default(self) -> None:
        model = ChatAnthropic(model="claude-test", api_key="test-key")
        assert model._client is not None

    def test_disables_sdk_retries_on_its_own_client(self) -> None:
        model = ChatAnthropic(model="claude-test", api_key="test-key")
        assert model._client.max_retries == 0
