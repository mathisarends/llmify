import inspect
import json
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal, Self, cast, overload

import httpx
from pydantic import BaseModel

try:
    from openai import AsyncOpenAI
    from openai.types.responses import (
        Response,
        ResponseCompletedEvent,
        ResponseErrorEvent,
        ResponseFailedEvent,
        ResponseFunctionToolCall,
        ResponseIncompleteEvent,
        ResponseOutputItem,
        ResponseOutputItemAddedEvent,
        ResponseOutputItemDoneEvent,
        ResponseReasoningSummaryTextDeltaEvent,
        ResponseTextDeltaEvent,
    )
except ImportError:
    raise ImportError(
        "The 'openai' package is required for Responses API providers. "
        "Install it with: pip install py-llmify[openai]"
    )

from llmify.base import ChatModel
from llmify.exceptions import LLMifyError, RateLimitError, RetryableError
from llmify.messages import (
    AssistantMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    Message,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from llmify.providers._openai_utils import (
    map_openai_error,
    reject_stream_parameter,
    resolve_api_key,
    tool_call,
    tool_schemas,
)
from llmify.retries import RetryCallback, retry_call, retry_stream
from llmify.tools import Tool, ToolChoice
from llmify.views import StreamTextDelta, StreamToolCall

from .transport import (
    HTTPResponsesTransport,
    ResponsesSession,
    ResponsesTransport,
    WebSocketResponsesTransport,
)
from .types import (
    ContinuationMode,
    OpenAIResponsesCompletion,
    OpenAIResponsesState,
    OpenAIResponsesStreamEnd,
    OpenAIResponsesStreamEvent,
    OpenAIResponsesUsage,
    PromptCacheOptions,
    ReasoningSummary,
    ResponsesOptions,
    StreamOutputItemAdded,
    StreamOutputItemDone,
    StreamReasoningSummaryDelta,
)

_CHAT_ONLY_PARAMS = frozenset(
    {"frequency_penalty", "presence_penalty", "stop", "seed", "response_format"}
)

type ReasoningEffort = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh", "max"
]
"""Reasoning effort levels accepted by current Responses models."""

type ToolExecutor = Callable[[ToolCall], object | Awaitable[object]]


class ChatOpenAIResponses(ChatModel):
    def __init__(
        self,
        model: str,
        api_key: str | Callable[[], Awaitable[str]] | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        store: bool = False,
        transport: ResponsesTransport | None = None,
        responses_options: ResponsesOptions | None = None,
        continuation_mode: ContinuationMode = ContinuationMode.STATELESS,
        preserve_reasoning: bool = True,
        reasoning_summary: ReasoningSummary | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_options: PromptCacheOptions | None = None,
        timeout: float | httpx.Timeout | None = 60.0,
        max_retries: int = 2,
        on_retry: RetryCallback | None = None,
        default_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        reject_stream_parameter(kwargs)
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        super().__init__(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            max_retries=max_retries,
            on_retry=on_retry,
            **kwargs,
        )
        api_key = self._resolve_api_key(api_key)

        self._api_key = api_key
        self._store = store
        self._transport = transport or HTTPResponsesTransport()
        self._responses_options = responses_options or ResponsesOptions(
            continuation_mode=continuation_mode,
            preserve_reasoning=preserve_reasoning,
            reasoning_summary=reasoning_summary,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_options=prompt_cache_options,
        )
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
            default_headers=default_headers,
        )

    @property
    def is_prewarmed(self) -> bool:
        """Whether this model has an open prewarmed WebSocket connection."""
        if not isinstance(self._transport, WebSocketResponsesTransport):
            return False
        return self._transport.is_prewarmed

    async def prewarm(self) -> None:
        """Open the configured WebSocket before a latency-critical request.

        No model request or input is sent. The connection stays available for
        sequential ``invoke()`` and ``stream()`` calls until ``aclose()`` or
        ``close_prewarmed()`` is called.
        """
        if not isinstance(self._transport, WebSocketResponsesTransport):
            raise LLMifyError("Prewarming requires WebSocketResponsesTransport.")
        await self._resolve_websocket_api_key()
        await self._transport.prewarm(self._client)

    async def close_prewarmed(self) -> None:
        """Close the retained WebSocket while keeping this model reusable."""
        if isinstance(self._transport, WebSocketResponsesTransport):
            await self._transport.aclose()

    async def aclose(self) -> None:
        """Close the prewarmed WebSocket and the underlying OpenAI client."""
        await self.close_prewarmed()
        await self._client.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    def _resolve_api_key(
        self,
        api_key: str | Callable[[], Awaitable[str]] | None,
    ) -> str | Callable[[], Awaitable[str]]:
        return resolve_api_key(api_key, "OPENAI_API_KEY", "OpenAI")

    @overload
    async def invoke[T: BaseModel](
        self,
        messages: list[Message],
        output_format: type[T],
        *,
        provider_state: OpenAIResponsesState | None = None,
        responses_options: ResponsesOptions | None = None,
        on_retry: RetryCallback | None = None,
        **kwargs: Any,
    ) -> OpenAIResponsesCompletion[T]: ...

    @overload
    async def invoke(
        self,
        messages: list[Message],
        output_format: None = None,
        *,
        provider_state: OpenAIResponsesState | None = None,
        responses_options: ResponsesOptions | None = None,
        on_retry: RetryCallback | None = None,
        **kwargs: Any,
    ) -> OpenAIResponsesCompletion[str]: ...

    async def invoke[T: BaseModel](
        self,
        messages: list[Message],
        output_format: type[T] | None = None,
        tools: list[Tool | dict] | None = None,
        tool_choice: ToolChoice = "auto",
        provider_state: OpenAIResponsesState | None = None,
        responses_options: ResponsesOptions | None = None,
        on_retry: RetryCallback | None = None,
        **kwargs: Any,
    ) -> OpenAIResponsesCompletion[T] | OpenAIResponsesCompletion[str]:
        reject_stream_parameter(kwargs)
        options = responses_options or self._responses_options
        async with self._transport_session() as session:
            end = await self._collect(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                provider_state=provider_state,
                options=options,
                params=_responses_params(self._merge_params(kwargs)),
                text=_json_schema_format(output_format),
                on_retry=on_retry if on_retry is not None else self._on_retry,
                session=session,
            )
        return _completion_from_end(end, output_format)

    async def invoke_with_tools[T: BaseModel](
        self,
        messages: list[Message],
        tools: list[Tool | dict],
        output_format: type[T] | None = None,
        *,
        tool_choice: ToolChoice = "auto",
        tool_executor: ToolExecutor | None = None,
        max_tool_rounds: int = 8,
        provider_state: OpenAIResponsesState | None = None,
        responses_options: ResponsesOptions | None = None,
        on_retry: RetryCallback | None = None,
        **kwargs: Any,
    ) -> OpenAIResponsesCompletion[T] | OpenAIResponsesCompletion[str]:
        """Run function calls to completion while preserving every native item.

        FunctionTool instances execute directly. Dict and RawSchemaTool entries
        require ``tool_executor`` because they do not contain an implementation.
        Tool exceptions are serialized as function-call outputs so the model can
        recover. ``max_tool_rounds`` bounds model/tool round trips.
        """
        if max_tool_rounds < 0:
            raise ValueError("'max_tool_rounds' must be greater than or equal to 0.")
        reject_stream_parameter(kwargs)

        options = responses_options or self._responses_options
        params = _responses_params(self._merge_params(kwargs))
        state = provider_state
        next_messages: list[Message] = messages
        all_tool_calls: list[ToolCall] = []
        total_usage: OpenAIResponsesUsage | None = None

        async with self._transport_session() as session:
            for round_index in range(max_tool_rounds + 1):
                end = await self._collect(
                    next_messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    provider_state=state,
                    options=options,
                    params=params,
                    text=_json_schema_format(output_format),
                    on_retry=on_retry if on_retry is not None else self._on_retry,
                    session=session,
                )
                state = end.provider_state
                total_usage = _add_usage(total_usage, end.usage)
                all_tool_calls.extend(end.tool_calls)

                if not end.tool_calls:
                    completed = _completion_from_end(end, output_format)
                    completed.tool_calls = all_tool_calls
                    completed.usage = total_usage
                    return completed

                if round_index == max_tool_rounds:
                    raise LLMifyError(
                        f"Tool loop exceeded max_tool_rounds={max_tool_rounds}."
                    )

                outputs = await _execute_tool_calls(
                    end.tool_calls,
                    tools,
                    executor=tool_executor,
                )
                next_messages = [
                    ToolResultMessage(tool_call_id=call.id, content=output)
                    for call, output in zip(end.tool_calls, outputs, strict=True)
                ]

        raise AssertionError("unreachable")

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool | dict] | None = None,
        tool_choice: ToolChoice = "auto",
        provider_state: OpenAIResponsesState | None = None,
        responses_options: ResponsesOptions | None = None,
        on_retry: RetryCallback | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        reject_stream_parameter(kwargs)
        async for event in self._stream(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            provider_state=provider_state,
            options=responses_options or self._responses_options,
            params=_responses_params(self._merge_params(kwargs)),
            on_retry=on_retry if on_retry is not None else self._on_retry,
        ):
            yield event

    async def _collect(
        self,
        messages: list[Message],
        tools: list[Tool | dict] | None,
        tool_choice: ToolChoice,
        provider_state: OpenAIResponsesState | None,
        options: ResponsesOptions,
        params: dict[str, Any],
        text: dict[str, Any] | None,
        on_retry: RetryCallback | None,
        session: ResponsesSession,
    ) -> OpenAIResponsesStreamEnd:
        async def collect_once() -> OpenAIResponsesStreamEnd:
            end: OpenAIResponsesStreamEnd | None = None
            async for event in self._stream_once(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                provider_state=provider_state,
                options=options,
                params=params,
                text=text,
                session=session,
            ):
                if isinstance(event, OpenAIResponsesStreamEnd):
                    end = event
            if end is None:
                raise LLMifyError(
                    "The Responses stream ended without a terminal event."
                )
            return end

        return await retry_call(
            collect_once,
            max_retries=self._default_max_retries,
            on_retry=on_retry,
            map_error=map_openai_error,
        )

    async def _stream(
        self,
        messages: list[Message],
        tools: list[Tool | dict] | None,
        tool_choice: ToolChoice,
        provider_state: OpenAIResponsesState | None,
        options: ResponsesOptions,
        params: dict[str, Any],
        text: dict[str, Any] | None = None,
        on_retry: RetryCallback | None = None,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        async with self._transport_session() as session:
            async for event in retry_stream(
                lambda: self._stream_once(
                    messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    provider_state=provider_state,
                    options=options,
                    params=params,
                    text=text,
                    session=session,
                ),
                max_retries=self._default_max_retries,
                on_retry=on_retry,
                map_error=map_openai_error,
            ):
                yield event

    @asynccontextmanager
    async def _transport_session(self) -> AsyncGenerator[ResponsesSession, None]:
        if isinstance(self._transport, WebSocketResponsesTransport):
            await self._resolve_websocket_api_key()

        async with self._transport.session(self._client) as session:
            yield session

    async def _resolve_websocket_api_key(self) -> None:
        if callable(self._api_key):
            self._client.api_key = await self._api_key()

    async def _stream_once(
        self,
        messages: list[Message],
        tools: list[Tool | dict] | None,
        tool_choice: ToolChoice,
        provider_state: OpenAIResponsesState | None,
        options: ResponsesOptions,
        params: dict[str, Any],
        session: ResponsesSession,
        text: dict[str, Any] | None = None,
    ) -> AsyncIterator[OpenAIResponsesStreamEvent]:
        if (
            provider_state is not None
            and provider_state.continuation_mode != options.continuation_mode
        ):
            raise ValueError(
                "provider_state.continuation_mode does not match ResponsesOptions."
            )

        previous_id = provider_state.response_id if provider_state is not None else None
        can_continue = bool(
            options.continuation_mode == ContinuationMode.PREVIOUS_RESPONSE_ID
            and previous_id
            and (self._store or session.can_continue_from(previous_id))
        )
        request, new_input_items, instructions = _build_request(
            model=self._model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            state=provider_state,
            options=options,
            params=params,
            text=text,
            store=self._store,
            can_continue=can_continue,
        )
        previous_items = (
            provider_state.input_items if provider_state is not None else []
        )

        text_acc: list[str] = []
        summary_acc: list[str] = []
        output_items: list[dict[str, Any]] = []
        tool_calls: list[ToolCall] = []
        usage: OpenAIResponsesUsage | None = None
        stop_reason: str | None = None
        response_id: str | None = None

        async for event in session.events(request):
            if isinstance(event, ResponseTextDeltaEvent):
                text_acc.append(event.delta)
                yield StreamTextDelta(delta=event.delta)

            elif isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
                summary_acc.append(event.delta)
                yield StreamReasoningSummaryDelta(delta=event.delta)

            elif isinstance(event, ResponseOutputItemAddedEvent):
                yield StreamOutputItemAdded(
                    output_index=event.output_index,
                    item=_dump_item(event.item),
                )

            elif isinstance(event, ResponseOutputItemDoneEvent):
                item = _dump_item(event.item)
                output_items.append(item)
                parsed_call = _parse_function_call(event.item)
                if parsed_call is not None:
                    tool_calls.append(parsed_call)
                    yield StreamToolCall(tool_call=parsed_call)
                yield StreamOutputItemDone(
                    output_index=event.output_index,
                    item=item,
                )

            elif isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent)):
                usage = _parse_responses_usage(event.response.usage)
                stop_reason = _parse_stop_reason(event.response, tool_calls)
                response_id = getattr(event.response, "id", None)
                output_items = _merge_response_output(output_items, event.response)

            elif isinstance(event, (ResponseFailedEvent, ResponseErrorEvent)):
                raise _event_error(event)

        full_input_items = [*previous_items, *new_input_items, *output_items]
        state = OpenAIResponsesState(
            continuation_mode=options.continuation_mode,
            input_items=full_input_items,
            output_items=output_items,
            response_id=response_id,
            instructions=instructions,
        )
        if response_id is not None:
            session.remember(response_id)

        yield OpenAIResponsesStreamEnd(
            stop_reason=stop_reason,
            usage=usage,
            tool_calls=tool_calls,
            completion="".join(text_acc),
            reasoning_summary="".join(summary_acc) or None,
            provider_state=state,
        )


def _completion_from_end[T: BaseModel](
    end: OpenAIResponsesStreamEnd,
    output_format: type[T] | None,
) -> OpenAIResponsesCompletion[T] | OpenAIResponsesCompletion[str]:
    if output_format is None:
        completion: T | str = end.completion
    else:
        try:
            completion = output_format.model_validate_json(end.completion)
        except ValueError as exc:
            raise LLMifyError(
                f"Model did not return valid {output_format.__name__} JSON: {exc}"
            ) from exc

    return OpenAIResponsesCompletion(
        completion=completion,
        thinking=end.reasoning_summary,
        reasoning_summary=end.reasoning_summary,
        stop_reason=end.stop_reason,
        usage=end.usage,
        tool_calls=end.tool_calls,
        provider_state=end.provider_state,
    )


async def _execute_tool_calls(
    calls: list[ToolCall],
    tools: list[Tool | dict],
    *,
    executor: ToolExecutor | None,
) -> list[str]:
    by_name = {
        tool.name: tool
        for tool in tools
        if not isinstance(tool, dict) and hasattr(tool, "name")
    }
    outputs: list[str] = []
    for call in calls:
        try:
            if executor is not None:
                result = executor(call)
            else:
                tool = by_name.get(call.function.name)
                if tool is None or not callable(tool):
                    raise LookupError(
                        f"No executable tool named {call.function.name!r}."
                    )
                arguments = tool.parse_arguments(call.function.arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("Function tool arguments must decode to an object.")
                result = cast(Callable[..., object], tool)(**arguments)
            if inspect.isawaitable(result):
                result = await result
            outputs.append(_serialize_tool_output(result))
        except Exception as exc:  # noqa: BLE001 - failures become tool outputs
            outputs.append(
                json.dumps(
                    {
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    },
                    ensure_ascii=False,
                )
            )
    return outputs


def _serialize_tool_output(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value, ensure_ascii=False, default=str)


def _build_request(
    *,
    model: str,
    messages: list[Message],
    tools: list[Tool | dict] | None,
    tool_choice: ToolChoice,
    state: OpenAIResponsesState | None,
    options: ResponsesOptions,
    params: dict[str, Any],
    text: dict[str, Any] | None,
    store: bool,
    can_continue: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    """Build a Responses request without transport or streaming side effects."""
    current_instructions, new_input_items = _convert_messages(messages)
    instructions = current_instructions or (
        state.instructions if state is not None else None
    )
    previous_items = state.input_items if state is not None else []
    request_input = (
        new_input_items if can_continue else [*previous_items, *new_input_items]
    )

    request: dict[str, Any] = {
        "model": model,
        "input": request_input,
        "store": store,
        **params,
    }
    if can_continue and state is not None and state.response_id is not None:
        request["previous_response_id"] = state.response_id
    if instructions:
        request["instructions"] = instructions
    if text is not None:
        request["text"] = text
    if tools:
        request["tools"] = _convert_tools(tools)
        request["tool_choice"] = tool_choice
    if options.prompt_cache_key is not None:
        request["prompt_cache_key"] = options.prompt_cache_key
    if options.prompt_cache_options is not None:
        request["prompt_cache_options"] = options.prompt_cache_options.model_dump(
            exclude_none=True
        )
    if options.reasoning_summary is not None:
        request["reasoning"] = {
            **request.get("reasoning", {}),
            "summary": options.reasoning_summary,
        }
    if options.preserve_reasoning and not store and not can_continue:
        includes = list(request.get("include") or [])
        if "reasoning.encrypted_content" not in includes:
            includes.append("reasoning.encrypted_content")
        request["include"] = includes

    return request, new_input_items, instructions


def _user_content(msg: UserMessage) -> str | list[dict]:
    if isinstance(msg.content, str):
        if not msg.cache:
            return msg.content
        return [_input_text(msg.content, cache=True)]

    content = []
    last_index = len(msg.content) - 1
    for index, part in enumerate(msg.content):
        cache = msg.cache and index == last_index
        if isinstance(part, ContentPartTextParam):
            content.append(_input_text(part.text, cache=cache))
        elif isinstance(part, ContentPartImageParam):
            image: dict[str, Any] = {
                "type": "input_image",
                "image_url": part.image_url.url,
                "detail": part.image_url.detail,
            }
            if cache:
                image["prompt_cache_breakpoint"] = {"mode": "explicit"}
            content.append(image)
    return content


def _input_text(text: str, *, cache: bool) -> dict[str, Any]:
    part: dict[str, Any] = {"type": "input_text", "text": text}
    if cache:
        part["prompt_cache_breakpoint"] = {"mode": "explicit"}
    return part


def _convert_messages(messages: list[Message]) -> tuple[str | None, list[dict]]:
    instructions: list[str] = []
    items: list[dict] = []

    for message in messages:
        if isinstance(message, SystemMessage):
            if not message.text:
                continue
            if message.cache:
                items.append(
                    {
                        "role": "developer",
                        "content": [_input_text(message.text, cache=True)],
                    }
                )
            else:
                instructions.append(message.text)
        elif isinstance(message, UserMessage):
            items.append({"role": "user", "content": _user_content(message)})
        elif isinstance(message, AssistantMessage):
            if message.text:
                items.append({"role": "assistant", "content": message.text})
            for call in message.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                )
        elif isinstance(message, ToolResultMessage):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )

    return "\n\n".join(instructions) or None, items


def _convert_tools(tools: list[Tool | dict]) -> list[dict]:
    converted = []
    for schema in tool_schemas(tools):
        function = schema.get("function", schema)
        converted.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function.get("description") or None,
                "parameters": function.get("parameters")
                or {"type": "object", "properties": {}},
                "strict": False,
            }
        )
    return converted


def _responses_params(params: dict[str, Any]) -> dict[str, Any]:
    for key in _CHAT_ONLY_PARAMS:
        params.pop(key, None)

    max_tokens = params.pop("max_tokens", None)
    if max_tokens is not None:
        params["max_output_tokens"] = max_tokens

    reasoning_effort = params.pop("reasoning_effort", None)
    if reasoning_effort is not None:
        params["reasoning"] = {
            **params.get("reasoning", {}),
            "effort": reasoning_effort,
        }

    return params


def _json_schema_format(output_format: type[BaseModel] | None) -> dict[str, Any] | None:
    if output_format is None:
        return None
    return {
        "format": {
            "type": "json_schema",
            "name": "output",
            "schema": _to_strict_schema(output_format.model_json_schema()),
            "strict": True,
        }
    }


def _to_strict_schema(schema: Any) -> Any:
    """Make a Pydantic JSON schema satisfy strict json_schema rules."""
    if isinstance(schema, list):
        return [_to_strict_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    strict = {key: _to_strict_schema(value) for key, value in schema.items()}
    if "properties" in strict:
        strict["additionalProperties"] = False
        strict["required"] = list(strict["properties"])
    return strict


def _dump_item(item: ResponseOutputItem) -> dict[str, Any]:
    return item.model_dump(mode="json", exclude_none=True)


def _merge_response_output(
    output_items: list[dict[str, Any]], response: Response
) -> list[dict[str, Any]]:
    seen_ids = {item.get("id") for item in output_items if item.get("id")}
    merged = list(output_items)
    for raw_item in getattr(response, "output", None) or []:
        item = _dump_item(raw_item)
        item_id = item.get("id")
        if item_id and item_id in seen_ids:
            continue
        if not item_id and item in merged:
            continue
        merged.append(item)
        if item_id:
            seen_ids.add(item_id)
    return merged


def _parse_function_call(item: ResponseOutputItem) -> ToolCall | None:
    if not isinstance(item, ResponseFunctionToolCall):
        return None
    return tool_call(
        call_id=item.call_id,
        name=item.name,
        arguments=item.arguments,
    )


def _parse_responses_usage(usage: Any | None) -> OpenAIResponsesUsage | None:
    if usage is None:
        return None
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return OpenAIResponsesUsage(
        prompt_tokens=usage.input_tokens,
        prompt_cached_tokens=getattr(input_details, "cached_tokens", None),
        prompt_cache_write_tokens=getattr(input_details, "cache_write_tokens", None),
        completion_tokens=usage.output_tokens,
        reasoning_tokens=getattr(output_details, "reasoning_tokens", None),
        total_tokens=usage.total_tokens,
    )


def _add_usage(
    left: OpenAIResponsesUsage | None,
    right: OpenAIResponsesUsage | None,
) -> OpenAIResponsesUsage | None:
    if left is None:
        return right
    if right is None:
        return left

    def add_optional(a: int | None, b: int | None) -> int | None:
        return None if a is None and b is None else (a or 0) + (b or 0)

    return OpenAIResponsesUsage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        prompt_cached_tokens=add_optional(
            left.prompt_cached_tokens, right.prompt_cached_tokens
        ),
        prompt_cache_write_tokens=add_optional(
            left.prompt_cache_write_tokens, right.prompt_cache_write_tokens
        ),
        completion_tokens=left.completion_tokens + right.completion_tokens,
        reasoning_tokens=add_optional(left.reasoning_tokens, right.reasoning_tokens),
        total_tokens=left.total_tokens + right.total_tokens,
    )


def _parse_stop_reason(response: Response, tool_calls: list[ToolCall]) -> str:
    if response.incomplete_details is not None:
        return response.incomplete_details.reason or "incomplete"
    if tool_calls:
        return "tool_calls"
    return response.status or "completed"


def _error_message(event: ResponseFailedEvent | ResponseErrorEvent) -> str:
    if isinstance(event, ResponseErrorEvent):
        return event.message
    if event.response.error is None:
        return "The Responses API reported an unspecified error."
    return event.response.error.message


def _event_error(
    event: ResponseFailedEvent | ResponseErrorEvent,
) -> LLMifyError:
    if isinstance(event, ResponseErrorEvent):
        code = event.code
    elif event.response.error is not None:
        code = event.response.error.code
    else:
        code = None

    message = _error_message(event)
    if code == "rate_limit_exceeded":
        return RateLimitError(message)
    if code in {"server_error", "vector_store_timeout"}:
        return RetryableError(message)
    return LLMifyError(message)
