import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal, overload

import httpx
from pydantic import BaseModel

try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError(
        "The 'openai' package is required for OpenAIResponses. "
        "Install it with: pip install py-llmify[openai]"
    )

from llmify.base import ChatModel
from llmify.exceptions import CredentialsUnavailableError, LLMifyError
from llmify.messages import (
    AssistantMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    Function,
    Message,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from llmify.providers.openai_compatible import _map_openai_error
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


class OpenAIResponses(ChatModel):
    """ChatModel backed by the OpenAI Responses API (`POST /responses`).

    Use this instead of `ChatOpenAI` for endpoints that only speak the Responses
    API, such as the Codex backend. Requests are always sent with `stream=True`
    because those backends reject unstreamed calls; `invoke` consumes the stream
    and returns the aggregated result.
    """

    def __init__(
        self,
        model: str,
        api_key: str | Callable[[], Awaitable[str]] | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        store: bool = False,
        timeout: float | httpx.Timeout | None = 60.0,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise CredentialsUnavailableError(
                "No OpenAI API key found. Pass 'api_key' or set OPENAI_API_KEY."
            )

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
        end = await self._collect(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            params=self._responses_params(kwargs),
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
        async for event in self._stream(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            params=self._responses_params(kwargs),
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
        instructions, input_items = self._convert_messages(messages)

        request: dict[str, Any] = {
            "model": self._model,
            "input": input_items,
            "store": self._store,
            "stream": True,
            **params,
        }
        if instructions:
            request["instructions"] = instructions
        if text is not None:
            request["text"] = text
        if tools:
            request["tools"] = self._convert_tools(tools)
            request["tool_choice"] = tool_choice

        try:
            stream = await self._client.responses.create(**request)
        except Exception as exc:
            mapped = _map_openai_error(exc)
            if mapped is not exc:
                raise mapped from exc
            raise

        text_acc: list[str] = []
        tool_calls: list[ToolCall] = []
        usage: ChatInvokeUsage | None = None
        stop_reason: str | None = None

        try:
            async for event in stream:
                event_type = getattr(event, "type", None)

                if event_type == "response.output_text.delta":
                    text_acc.append(event.delta)
                    yield StreamTextDelta(delta=event.delta)

                elif event_type == "response.output_item.done":
                    tool_call = _parse_function_call(event.item)
                    if tool_call is not None:
                        tool_calls.append(tool_call)
                        yield StreamToolCall(tool_call=tool_call)

                elif event_type in ("response.completed", "response.incomplete"):
                    usage = _parse_usage(getattr(event.response, "usage", None))
                    stop_reason = _parse_stop_reason(event.response, tool_calls)

                elif event_type in ("response.failed", "error"):
                    raise LLMifyError(_error_message(event))
        except Exception as exc:
            mapped = _map_openai_error(exc)
            if mapped is not exc:
                raise mapped from exc
            raise

        yield StreamEnd(
            stop_reason=stop_reason,
            usage=usage,
            tool_calls=tool_calls,
            completion="".join(text_acc),
        )

    def _responses_params(self, method_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Merge defaults and translate the chat-completions naming this library uses."""
        params = self._merge_params(method_kwargs)
        for key in _CHAT_ONLY_PARAMS:
            params.pop(key, None)

        max_tokens = params.pop("max_tokens", None)
        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens

        return params

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict]]:
        """Split messages into `instructions` and Responses API input items."""
        instructions: list[str] = []
        items: list[dict] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                if msg.text:
                    instructions.append(msg.text)
            elif isinstance(msg, UserMessage):
                items.append({"role": "user", "content": _user_content(msg)})
            elif isinstance(msg, AssistantMessage):
                if msg.text:
                    items.append({"role": "assistant", "content": msg.text})
                for tool_call in msg.tool_calls:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    )
            elif isinstance(msg, ToolResultMessage):
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id,
                        "output": msg.content,
                    }
                )

        return "\n\n".join(instructions) or None, items

    def _convert_tools(self, tools: list[Tool | dict]) -> list[dict]:
        """Flatten chat-style tool schemas into the Responses API's flat shape."""
        converted = []
        for tool in tools:
            schema = tool if isinstance(tool, dict) else tool.to_openai_schema()
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


def _parse_function_call(item: Any) -> ToolCall | None:
    if getattr(item, "type", None) != "function_call":
        return None
    return ToolCall(
        id=getattr(item, "call_id", None) or getattr(item, "id", ""),
        function=Function(
            name=item.name,
            arguments=item.arguments or "{}",
        ),
    )


def _parse_usage(usage: Any) -> ChatInvokeUsage | None:
    if not usage:
        return None
    return ChatInvokeUsage(
        prompt_tokens=usage.input_tokens,
        prompt_cached_tokens=getattr(
            getattr(usage, "input_tokens_details", None), "cached_tokens", None
        ),
        completion_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )


def _parse_stop_reason(response: Any, tool_calls: list[ToolCall]) -> str:
    incomplete_details = getattr(response, "incomplete_details", None)
    if incomplete_details is not None:
        return getattr(incomplete_details, "reason", None) or "incomplete"
    if tool_calls:
        return "tool_calls"
    return getattr(response, "status", None) or "completed"


def _error_message(event: Any) -> str:
    error = getattr(event, "error", None) or getattr(
        getattr(event, "response", None), "error", None
    )
    if error is None:
        return "The Responses API reported an unspecified error."
    message = getattr(error, "message", None)
    return message or json.dumps(error, default=str)
