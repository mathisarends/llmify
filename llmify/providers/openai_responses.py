from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal, overload

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
        ResponseOutputItemDoneEvent,
        ResponseTextDeltaEvent,
    )
except ImportError:
    raise ImportError(
        "The 'openai' package is required for Responses API providers. "
        "Install it with: pip install py-llmify[openai]"
    )

from llmify.base import ChatModel
from llmify.exceptions import LLMifyError
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
    openai_errors,
    parse_usage,
    reject_stream_parameter,
    resolve_api_key,
    tool_call,
    tool_schemas,
)
from llmify.tools import Tool
from llmify.views import (
    ChatInvokeCompletion,
    ChatInvokeUsage,
    StreamEnd,
    StreamEvent,
    StreamTextDelta,
    StreamToolCall,
)

_CHAT_ONLY_PARAMS = frozenset(
    {"frequency_penalty", "presence_penalty", "stop", "seed", "response_format"}
)

type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
"""Effort levels the Responses API accepts.

Not every model supports every level — `xhigh` is limited to the newest
reasoning models, and `none`/`minimal` are rejected by some. The API reports an
unsupported level as a request error.
"""


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
        timeout: float | httpx.Timeout | None = 60.0,
        max_retries: int = 2,
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
            **kwargs,
        )
        api_key = resolve_api_key(api_key, "OPENAI_API_KEY", "OpenAI")

        self._store = store
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )

    @overload
    async def invoke[T: BaseModel](
        self, messages: list[Message], output_format: type[T], **kwargs: Any
    ) -> ChatInvokeCompletion[T]: ...

    @overload
    async def invoke(
        self, messages: list[Message], output_format: None = None, **kwargs: Any
    ) -> ChatInvokeCompletion[str]: ...

    async def invoke[T: BaseModel](
        self,
        messages: list[Message],
        output_format: type[T] | None = None,
        tools: list[Tool | dict] | None = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        **kwargs: Any,
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        reject_stream_parameter(kwargs)
        end = await self._collect(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            params=_responses_params(self._merge_params(kwargs)),
            text=_json_schema_format(output_format),
        )

        if output_format is None:
            return ChatInvokeCompletion(
                completion=end.completion,
                stop_reason=end.stop_reason,
                usage=end.usage,
                tool_calls=end.tool_calls,
            )

        try:
            parsed = output_format.model_validate_json(end.completion)
        except ValueError as exc:
            raise LLMifyError(
                f"Model did not return valid {output_format.__name__} JSON: {exc}"
            ) from exc

        return ChatInvokeCompletion(
            completion=parsed,
            stop_reason=end.stop_reason,
            usage=end.usage,
            tool_calls=end.tool_calls,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool | dict] | None = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        reject_stream_parameter(kwargs)
        async for event in self._stream(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            params=_responses_params(self._merge_params(kwargs)),
        ):
            yield event

    async def _collect(
        self,
        messages: list[Message],
        tools: list[Tool | dict] | None,
        tool_choice: Literal["auto", "required", "none"],
        params: dict[str, Any],
        text: dict[str, Any] | None,
    ) -> StreamEnd:
        end = StreamEnd()
        async for event in self._stream(
            messages, tools=tools, tool_choice=tool_choice, params=params, text=text
        ):
            if isinstance(event, StreamEnd):
                end = event
        return end

    async def _stream(
        self,
        messages: list[Message],
        tools: list[Tool | dict] | None,
        tool_choice: Literal["auto", "required", "none"],
        params: dict[str, Any],
        text: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        instructions, input_items = _convert_messages(messages)

        request: dict[str, Any] = {
            "model": self._model,
            "input": input_items,
            "store": self._store,
            **params,
            "stream": True,
        }
        if instructions:
            request["instructions"] = instructions
        if text is not None:
            request["text"] = text
        if tools:
            request["tools"] = _convert_tools(tools)
            request["tool_choice"] = tool_choice

        with openai_errors():
            stream = await self._client.responses.create(**request)

        text_acc: list[str] = []
        tool_calls: list[ToolCall] = []
        usage: ChatInvokeUsage | None = None
        stop_reason: str | None = None

        with openai_errors():
            async for event in stream:
                if isinstance(event, ResponseTextDeltaEvent):
                    text_acc.append(event.delta)
                    yield StreamTextDelta(delta=event.delta)

                elif isinstance(event, ResponseOutputItemDoneEvent):
                    tool_call = _parse_function_call(event.item)
                    if tool_call is not None:
                        tool_calls.append(tool_call)
                        yield StreamToolCall(tool_call=tool_call)

                elif isinstance(
                    event, (ResponseCompletedEvent, ResponseIncompleteEvent)
                ):
                    usage = parse_usage(event.response.usage)
                    stop_reason = _parse_stop_reason(event.response, tool_calls)

                elif isinstance(event, (ResponseFailedEvent, ResponseErrorEvent)):
                    raise LLMifyError(_error_message(event))

        yield StreamEnd(
            stop_reason=stop_reason,
            usage=usage,
            tool_calls=tool_calls,
            completion="".join(text_acc),
        )


def _user_content(msg: UserMessage) -> str | list[dict]:
    if isinstance(msg.content, str):
        return msg.content

    content = []
    for part in msg.content:
        if isinstance(part, ContentPartTextParam):
            content.append({"type": "input_text", "text": part.text})
        elif isinstance(part, ContentPartImageParam):
            content.append(
                {
                    "type": "input_image",
                    "image_url": part.image_url.url,
                    "detail": part.image_url.detail,
                }
            )
    return content


def _convert_messages(messages: list[Message]) -> tuple[str | None, list[dict]]:
    instructions: list[str] = []
    items: list[dict] = []

    for message in messages:
        if isinstance(message, SystemMessage):
            if message.text:
                instructions.append(message.text)
        elif isinstance(message, UserMessage):
            items.append({"role": "user", "content": _user_content(message)})
        elif isinstance(message, AssistantMessage):
            if message.text:
                items.append({"role": "assistant", "content": message.text})
            for tool_call in message.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
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
    """Make a Pydantic JSON schema satisfy the API's strict json_schema rules.

    Every object must forbid additional properties and list all of its properties
    as required — Pydantic emits neither for optional fields.
    """
    if isinstance(schema, list):
        return [_to_strict_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    strict = {key: _to_strict_schema(value) for key, value in schema.items()}
    if "properties" in strict:
        strict["additionalProperties"] = False
        strict["required"] = list(strict["properties"])
    return strict


def _parse_function_call(item: ResponseOutputItem) -> ToolCall | None:
    if not isinstance(item, ResponseFunctionToolCall):
        return None
    return tool_call(
        call_id=item.call_id,
        name=item.name,
        arguments=item.arguments,
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
