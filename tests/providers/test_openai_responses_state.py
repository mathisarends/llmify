from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("openai")

from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningItem,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)

from llmify import (
    ChatOpenAIResponses,
    ContinuationMode,
    OpenAIResponsesState,
    PromptCacheOptions,
    ResponsesOptions,
    StreamOutputItemAdded,
    StreamOutputItemDone,
    StreamReasoningSummaryDelta,
    SystemMessage,
    UserMessage,
    WebSocketResponsesTransport,
    tool,
)
from llmify.providers.openai_responses import _build_request


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def _response(
    response_id: str,
    *,
    usage: ResponseUsage | None = None,
    output: list | None = None,
) -> Response:
    return Response.model_construct(
        id=response_id,
        status="completed",
        incomplete_details=None,
        error=None,
        usage=usage,
        output=output or [],
    )


def _completed(response: Response, sequence_number: int) -> ResponseCompletedEvent:
    return ResponseCompletedEvent.model_construct(
        response=response,
        sequence_number=sequence_number,
        type="response.completed",
    )


def _done(item, output_index: int, sequence_number: int):
    return ResponseOutputItemDoneEvent.model_construct(
        item=item,
        output_index=output_index,
        sequence_number=sequence_number,
        type="response.output_item.done",
    )


async def _stream(*events):
    for event in events:
        yield event


def _function_call(call_id: str, name: str, arguments: str):
    return ResponseFunctionToolCall.model_construct(
        type="function_call",
        call_id=call_id,
        id=f"item_{call_id}",
        name=name,
        arguments=arguments,
        status="completed",
    )


class TestNativeState:
    @pytest.mark.asyncio
    async def test_collects_every_done_item_and_replays_encrypted_reasoning(
        self,
    ) -> None:
        reasoning = ResponseReasoningItem.model_construct(
            id="reasoning_1",
            type="reasoning",
            summary=[],
            encrypted_content="opaque-ciphertext",
            status="completed",
        )
        call = _function_call("call_1", "lookup", '{"query":"x"}')
        model = ChatOpenAIResponses(model="gpt-test", store=False)
        model._client.responses.create = AsyncMock(
            side_effect=[
                _stream(
                    _done(reasoning, 0, 0),
                    _done(call, 1, 1),
                    _completed(_response("resp_1"), 2),
                ),
                _stream(_completed(_response("resp_2"), 0)),
            ]
        )

        first = await model.invoke([UserMessage(content="Start")])

        assert isinstance(first.provider_state, OpenAIResponsesState)
        assert first.provider_state.response_id == "resp_1"
        assert [item["type"] for item in first.provider_state.output_items] == [
            "reasoning",
            "function_call",
        ]
        assert (
            first.provider_state.output_items[0]["encrypted_content"]
            == "opaque-ciphertext"
        )
        first_request = model._client.responses.create.call_args_list[0].kwargs
        assert "reasoning.encrypted_content" in first_request["include"]

        await model.invoke(
            [UserMessage(content="Continue")],
            provider_state=first.provider_state,
        )

        second_request = model._client.responses.create.call_args_list[1].kwargs
        assert [item.get("type") for item in second_request["input"]] == [
            None,
            "reasoning",
            "function_call",
            None,
        ]
        assert second_request["input"][1]["encrypted_content"] == "opaque-ciphertext"

    @pytest.mark.asyncio
    async def test_previous_response_mode_sends_only_new_items_and_instructions(
        self,
    ) -> None:
        options = ResponsesOptions(
            continuation_mode=ContinuationMode.PREVIOUS_RESPONSE_ID
        )
        model = ChatOpenAIResponses(
            model="gpt-test",
            store=True,
            responses_options=options,
        )
        model._client.responses.create = AsyncMock(
            side_effect=[
                _stream(_completed(_response("resp_1"), 0)),
                _stream(_completed(_response("resp_2"), 0)),
            ]
        )

        first = await model.invoke(
            [SystemMessage(content="Stay concise"), UserMessage(content="One")]
        )
        await model.invoke(
            [UserMessage(content="Two")],
            provider_state=first.provider_state,
        )

        request = model._client.responses.create.call_args_list[1].kwargs
        assert request["previous_response_id"] == "resp_1"
        assert request["input"] == [{"role": "user", "content": "Two"}]
        assert request["instructions"] == "Stay concise"

    @pytest.mark.asyncio
    async def test_state_mode_mismatch_is_rejected(self) -> None:
        model = ChatOpenAIResponses(model="gpt-test")
        state = OpenAIResponsesState(
            continuation_mode=ContinuationMode.PREVIOUS_RESPONSE_ID
        )

        with pytest.raises(ValueError, match="continuation_mode"):
            await model.invoke([UserMessage(content="Hi")], provider_state=state)


class TestRequestBuilder:
    def test_builds_full_stateless_replay_without_side_effects(self) -> None:
        state = OpenAIResponsesState(
            input_items=[
                {"role": "user", "content": "Earlier"},
                {
                    "type": "reasoning",
                    "id": "reasoning_1",
                    "encrypted_content": "ciphertext",
                    "summary": [],
                },
            ],
            response_id="resp_1",
            instructions="Stable instructions",
        )

        request, new_items, instructions = _build_request(
            model="gpt-test",
            messages=[UserMessage(content="Now")],
            tools=None,
            tool_choice="auto",
            state=state,
            options=ResponsesOptions(),
            params={"max_output_tokens": 100},
            text=None,
            store=False,
            can_continue=False,
        )

        assert new_items == [{"role": "user", "content": "Now"}]
        assert instructions == "Stable instructions"
        assert request["input"] == [
            *state.input_items,
            {"role": "user", "content": "Now"},
        ]
        assert request["include"] == ["reasoning.encrypted_content"]
        assert "previous_response_id" not in request

    def test_builds_incremental_previous_response_request(self) -> None:
        state = OpenAIResponsesState(
            continuation_mode=ContinuationMode.PREVIOUS_RESPONSE_ID,
            input_items=[{"role": "user", "content": "Earlier"}],
            response_id="resp_1",
        )
        options = ResponsesOptions(
            continuation_mode=ContinuationMode.PREVIOUS_RESPONSE_ID
        )

        request, _, _ = _build_request(
            model="gpt-test",
            messages=[UserMessage(content="Now")],
            tools=None,
            tool_choice="auto",
            state=state,
            options=options,
            params={},
            text=None,
            store=False,
            can_continue=True,
        )

        assert request["previous_response_id"] == "resp_1"
        assert request["input"] == [{"role": "user", "content": "Now"}]
        assert "include" not in request


class TestResponsesOnlyStreaming:
    @pytest.mark.asyncio
    async def test_emits_native_item_and_reasoning_summary_events(self) -> None:
        reasoning = ResponseReasoningItem.model_construct(
            id="reasoning_1",
            type="reasoning",
            summary=[],
            encrypted_content="ciphertext",
            status="in_progress",
        )
        added = ResponseOutputItemAddedEvent.model_construct(
            item=reasoning,
            output_index=0,
            sequence_number=0,
            type="response.output_item.added",
        )
        summary = ResponseReasoningSummaryTextDeltaEvent.model_construct(
            delta="Checked the constraints.",
            item_id="reasoning_1",
            output_index=0,
            sequence_number=1,
            summary_index=0,
            type="response.reasoning_summary_text.delta",
        )
        done = _done(reasoning, 0, 2)
        model = ChatOpenAIResponses(
            model="gpt-test",
            reasoning_summary="auto",
        )
        model._client.responses.create = AsyncMock(
            return_value=_stream(
                added,
                summary,
                done,
                _completed(_response("resp_1"), 3),
            )
        )

        events = [event async for event in model.stream([UserMessage(content="Hi")])]

        assert isinstance(events[0], StreamOutputItemAdded)
        assert isinstance(events[1], StreamReasoningSummaryDelta)
        assert isinstance(events[2], StreamOutputItemDone)
        assert events[-1].reasoning_summary == "Checked the constraints."
        request = model._client.responses.create.call_args.kwargs
        assert request["reasoning"] == {"summary": "auto"}

    @pytest.mark.asyncio
    async def test_reports_cache_writes_and_reasoning_tokens(self) -> None:
        usage = ResponseUsage.model_construct(
            input_tokens=10,
            input_tokens_details=SimpleNamespace(
                cached_tokens=4,
                cache_write_tokens=3,
            ),
            output_tokens=8,
            output_tokens_details=SimpleNamespace(reasoning_tokens=6),
            total_tokens=18,
        )
        model = ChatOpenAIResponses(model="gpt-test")
        model._client.responses.create = AsyncMock(
            return_value=_stream(_completed(_response("resp_1", usage=usage), 0))
        )

        result = await model.invoke([UserMessage(content="Hi")])

        assert result.usage is not None
        assert result.usage.prompt_cached_tokens == 4
        assert result.usage.prompt_cache_write_tokens == 3
        assert result.usage.reasoning_tokens == 6


class TestPromptCaching:
    @pytest.mark.asyncio
    async def test_sends_key_options_and_explicit_message_breakpoint(self) -> None:
        options = ResponsesOptions(
            prompt_cache_key="session-123",
            prompt_cache_options=PromptCacheOptions(mode="explicit", ttl="30m"),
        )
        model = ChatOpenAIResponses(model="gpt-test", responses_options=options)
        model._client.responses.create = AsyncMock(
            return_value=_stream(_completed(_response("resp_1"), 0))
        )

        await model.invoke(
            [
                SystemMessage(content="Stable instructions", cache=True),
                UserMessage(content="Dynamic question"),
            ]
        )

        request = model._client.responses.create.call_args.kwargs
        assert request["prompt_cache_key"] == "session-123"
        assert request["extra_body"]["prompt_cache_options"] == {
            "mode": "explicit",
            "ttl": "30m",
        }
        assert "instructions" not in request
        assert request["input"][0] == {
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": "Stable instructions",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        }


class TestToolLoop:
    @pytest.mark.asyncio
    async def test_executes_multiple_calls_and_returns_failures_as_outputs(
        self,
    ) -> None:
        @tool
        def add(a: int, b: int) -> int:
            return a + b

        @tool
        def fail() -> str:
            raise RuntimeError("broken")

        text_delta = ResponseTextDeltaEvent.model_construct(
            content_index=0,
            delta="Finished",
            item_id="message_1",
            logprobs=[],
            output_index=0,
            sequence_number=0,
            type="response.output_text.delta",
        )
        model = ChatOpenAIResponses(model="gpt-test")
        model._client.responses.create = AsyncMock(
            side_effect=[
                _stream(
                    _done(_function_call("call_add", "add", '{"a":2,"b":3}'), 0, 0),
                    _done(_function_call("call_fail", "fail", "{}"), 1, 1),
                    _completed(_response("resp_1"), 2),
                ),
                _stream(text_delta, _completed(_response("resp_2"), 1)),
            ]
        )

        result = await model.invoke_with_tools(
            [UserMessage(content="Use both")],
            tools=[add, fail],
            max_tool_rounds=2,
        )

        assert result.completion == "Finished"
        assert [call.id for call in result.tool_calls] == ["call_add", "call_fail"]
        second_input = model._client.responses.create.call_args_list[1].kwargs["input"]
        outputs = [
            item for item in second_input if item.get("type") == "function_call_output"
        ]
        assert outputs[0]["output"] == "5"
        assert '"type": "RuntimeError"' in outputs[1]["output"]
        assert '"message": "broken"' in outputs[1]["output"]

    @pytest.mark.asyncio
    async def test_enforces_maximum_tool_rounds(self) -> None:
        @tool
        def again() -> str:
            return "again"

        model = ChatOpenAIResponses(model="gpt-test")
        model._client.responses.create = AsyncMock(
            return_value=_stream(
                _done(_function_call("call_1", "again", "{}"), 0, 0),
                _completed(_response("resp_1"), 1),
            )
        )

        with pytest.raises(Exception, match="max_tool_rounds=0"):
            await model.invoke_with_tools(
                [UserMessage(content="Loop")],
                tools=[again],
                max_tool_rounds=0,
            )


class _ConnectionManager:
    def __init__(self, connection) -> None:
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, *_args):
        return None


class TestWebSocketTransport:
    @pytest.mark.asyncio
    async def test_resolves_dynamic_auth_and_forwards_default_headers(self) -> None:
        api_key = AsyncMock(return_value="dynamic-token")
        connection = SimpleNamespace(
            send=AsyncMock(),
            recv=AsyncMock(side_effect=[_completed(_response("resp_1"), 0)]),
        )
        model = ChatOpenAIResponses(
            model="gpt-test",
            api_key=api_key,
            default_headers={"ChatGPT-Account-Id": "acct-123"},
            transport=WebSocketResponsesTransport(),
        )
        model._client.responses.connect = Mock(
            return_value=_ConnectionManager(connection)
        )

        await model.invoke([UserMessage(content="Hi")])

        api_key.assert_awaited_once_with()
        headers = model._client.responses.connect.call_args.kwargs["extra_headers"]
        assert headers["Authorization"] == "Bearer dynamic-token"
        assert headers["ChatGPT-Account-Id"] == "acct-123"

    @pytest.mark.asyncio
    async def test_uses_response_create_without_http_stream_fields(self) -> None:
        connection = SimpleNamespace(
            send=AsyncMock(),
            recv=AsyncMock(side_effect=[_completed(_response("resp_1"), 0)]),
        )
        model = ChatOpenAIResponses(
            model="gpt-test",
            transport=WebSocketResponsesTransport(),
        )
        model._client.responses.connect = Mock(
            return_value=_ConnectionManager(connection)
        )
        model._client.responses.create = AsyncMock()

        result = await model.invoke([UserMessage(content="Hi")])

        assert result.provider_state.response_id == "resp_1"
        request = connection.send.await_args.args[0]
        assert request["type"] == "response.create"
        assert "stream" not in request
        assert "background" not in request
        model._client.responses.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tool_loop_reuses_connection_and_previous_response_id(self) -> None:
        @tool
        def ping() -> str:
            return "pong"

        connection = SimpleNamespace(
            send=AsyncMock(),
            recv=AsyncMock(
                side_effect=[
                    _done(_function_call("call_1", "ping", "{}"), 0, 0),
                    _completed(_response("resp_1"), 1),
                    _completed(_response("resp_2"), 0),
                ]
            ),
        )
        options = ResponsesOptions(
            continuation_mode=ContinuationMode.PREVIOUS_RESPONSE_ID,
        )
        model = ChatOpenAIResponses(
            model="gpt-test",
            store=False,
            transport=WebSocketResponsesTransport(),
            responses_options=options,
        )
        model._client.responses.connect = Mock(
            return_value=_ConnectionManager(connection)
        )

        await model.invoke_with_tools(
            [UserMessage(content="Ping")],
            tools=[ping],
        )

        assert connection.send.await_count == 2
        second = connection.send.await_args_list[1].args[0]
        assert second["previous_response_id"] == "resp_1"
        assert second["input"] == [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "pong",
            }
        ]
