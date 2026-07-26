import json
import os
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any, Literal, overload

import httpx
from pydantic import BaseModel

try:
    from anthropic import (
        APIConnectionError as _AnthropicConnectionError,
    )
    from anthropic import (
        APIStatusError as _AnthropicStatusError,
    )
    from anthropic import (
        APITimeoutError as _AnthropicTimeoutError,
    )
    from anthropic import (
        AsyncAnthropic,
    )
    from anthropic import (
        RateLimitError as _AnthropicRateLimitError,
    )
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types import Usage as AnthropicUsage
except ImportError:
    raise ImportError(
        "The 'anthropic' package is required for ChatAnthropic. "
        "Install it with: pip install py-llmify[anthropic]"
    )


from llmify.base import ChatModel
from llmify.exceptions import (
    AuthenticationError,
    ContextLengthExceededError,
    OutOfCreditsError,
    RateLimitError,
    RetryableError,
)
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
from llmify.tools import Tool
from llmify.views import (
    ChatInvokeCompletion,
    ChatInvokeUsage,
    StreamEnd,
    StreamEvent,
    StreamTextDelta,
    StreamToolCall,
)


def _map_anthropic_error(exc: Exception) -> Exception:
    if isinstance(exc, _AnthropicRateLimitError):
        body = exc.body or {}
        error_type = (
            (body.get("error") or {}).get("type", "") if isinstance(body, dict) else ""
        )
        if error_type == "credit_balance_too_low":
            return OutOfCreditsError(str(exc))
        retry_after: float | None = None
        raw = exc.response.headers.get("retry-after")
        if raw:
            try:
                retry_after = float(raw)
            except ValueError:
                pass
        return RateLimitError(str(exc), retry_after=retry_after)
    if isinstance(exc, _AnthropicStatusError) and exc.status_code == 402:
        return OutOfCreditsError(str(exc))
    if isinstance(exc, _AnthropicStatusError) and exc.status_code == 400:
        body = exc.body or {}
        message = (
            (body.get("error") or {}).get("message", "")
            if isinstance(body, dict)
            else ""
        )
        msg_lower = message.lower()
        if (
            "too long" in msg_lower
            or "context" in msg_lower
            or (
                "token" in msg_lower
                and any(kw in msg_lower for kw in ("exceed", "maximum", "limit"))
            )
        ):
            return ContextLengthExceededError(str(exc))
    if isinstance(exc, _AnthropicStatusError) and exc.status_code == 401:
        return AuthenticationError(str(exc))
    if isinstance(exc, (_AnthropicConnectionError, _AnthropicTimeoutError)):
        return RetryableError(str(exc))
    if isinstance(exc, _AnthropicStatusError) and exc.status_code >= 500:
        return RetryableError(str(exc), status_code=exc.status_code)
    return exc


class AnthropicModel(StrEnum):
    CLAUDE_FABLE_5 = "claude-fable-5"
    CLAUDE_OPUS_4_8 = "claude-opus-4-8"
    CLAUDE_SONNET_5 = "claude-sonnet-5"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5"

    CLAUDE_OPUS_4_7 = "claude-opus-4-7"
    CLAUDE_OPUS_4_6 = "claude-opus-4-6"
    CLAUDE_SONNET_4_6 = "claude-sonnet-4-6"


def _build_params(
    model: str, messages: list[Message], merged: dict[str, Any]
) -> dict[str, Any]:
    system_text, converted = _convert_messages(messages)

    params: dict[str, Any] = {
        "model": model,
        "messages": converted,
        "max_tokens": merged.pop("max_tokens", 4096) or 4096,
    }

    if system_text:
        params["system"] = system_text

    if "temperature" in merged:
        params["temperature"] = merged.pop("temperature")
    if "top_p" in merged:
        params["top_p"] = merged.pop("top_p")
    if "stop" in merged:
        stop = merged.pop("stop")
        params["stop_sequences"] = [stop] if isinstance(stop, str) else stop

    # Remove OpenAI-specific params that Anthropic doesn't support.
    merged.pop("frequency_penalty", None)
    merged.pop("presence_penalty", None)
    merged.pop("seed", None)
    merged.pop("response_format", None)

    params.update(merged)
    return params


def _extract_text(response: AnthropicMessage) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def _parse_usage(usage: AnthropicUsage) -> ChatInvokeUsage:
    return ChatInvokeUsage(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.input_tokens + usage.output_tokens,
        prompt_cached_tokens=usage.cache_read_input_tokens,
        prompt_cache_creation_tokens=usage.cache_creation_input_tokens,
    )


def _parse_tool_calls(response: AnthropicMessage) -> list[ToolCall]:
    return [
        ToolCall(
            id=block.id,
            function=Function(
                name=block.name,
                arguments=json.dumps(block.input),
            ),
        )
        for block in response.content
        if block.type == "tool_use"
    ]


def _convert_tool(tool: Tool) -> dict[str, Any]:
    openai_schema = tool.to_openai_schema()
    function = openai_schema.get("function", openai_schema)
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "input_schema": function.get("parameters", {}),
    }


def _convert_tools(tools: list[Tool | dict]) -> list[dict]:
    return [tool if isinstance(tool, dict) else _convert_tool(tool) for tool in tools]


def _convert_messages(
    messages: list[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_text: str | None = None
    converted: list[dict[str, Any]] = []

    for message in messages:
        if isinstance(message, SystemMessage):
            system_text = message.text
            continue

        if isinstance(message, ToolResultMessage):
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": message.content,
                        }
                    ],
                }
            )
            continue

        if isinstance(message, AssistantMessage) and message.tool_calls:
            content: list[dict[str, Any]] = []
            if message.text:
                content.append({"type": "text", "text": message.text})
            for tool_call in message.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": json.loads(tool_call.function.arguments),
                    }
                )
            converted.append({"role": "assistant", "content": content})
            continue

        if isinstance(message, UserMessage) and isinstance(message.content, list):
            content_parts: list[dict[str, Any]] = []
            for part in message.content:
                if isinstance(part, ContentPartTextParam):
                    content_parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ContentPartImageParam):
                    url = part.image_url.url
                    if url.startswith("data:"):
                        media_type, _, data = url.partition(";base64,")
                        content_parts.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type.removeprefix("data:"),
                                    "data": data,
                                },
                            }
                        )
                    else:
                        content_parts.append(
                            {
                                "type": "image",
                                "source": {"type": "url", "url": url},
                            }
                        )
            converted.append({"role": "user", "content": content_parts})
            continue

        if isinstance(message, UserMessage):
            converted.append({"role": "user", "content": message.text})
        elif isinstance(message, AssistantMessage):
            converted.append({"role": "assistant", "content": message.text})

    return system_text, converted


class ChatAnthropic(ChatModel):
    _client: AsyncAnthropic
    _model: str

    def __init__(
        self,
        model: str | AnthropicModel = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | list[str] | None = None,
        seed: int | None = None,
        response_format: dict | None = None,
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
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            seed=seed,
            response_format=response_format,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")

        self._client = AsyncAnthropic(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers or {},
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
        try:
            params = _build_params(self._model, messages, self._merge_params(kwargs))

            if output_format is not None:
                return await self._invoke_with_structured_output(params, output_format)

            if tools:
                return await self._invoke_with_tools(params, tools, tool_choice)

            return await self._invoke_plain(params)
        except Exception as exc:
            mapped = _map_anthropic_error(exc)
            if mapped is not exc:
                raise mapped from exc
            raise

    async def _invoke_plain(self, params: dict[str, Any]) -> ChatInvokeCompletion[str]:
        response: AnthropicMessage = await self._client.messages.create(**params)
        return ChatInvokeCompletion(
            completion=_extract_text(response),
            stop_reason=response.stop_reason,
            usage=_parse_usage(response.usage),
        )

    async def _invoke_with_tools(
        self,
        params: dict[str, Any],
        tools: list[Tool | dict],
        tool_choice: Literal["auto", "required", "none"] = "auto",
    ) -> ChatInvokeCompletion[str]:
        anthropic_tools = _convert_tools(tools)
        tool_choice_map = {
            "auto": {"type": "auto"},
            "required": {"type": "any"},
            "none": {"type": "none"},
        }
        response: AnthropicMessage = await self._client.messages.create(
            **params,
            tools=anthropic_tools,
            tool_choice=tool_choice_map.get(tool_choice, {"type": "auto"}),
        )
        return ChatInvokeCompletion(
            completion=_extract_text(response),
            tool_calls=_parse_tool_calls(response),
            stop_reason=response.stop_reason,
            usage=_parse_usage(response.usage),
        )

    async def _invoke_with_structured_output[T: BaseModel](
        self, params: dict[str, Any], output_format: type[T]
    ) -> ChatInvokeCompletion[T]:
        schema = output_format.model_json_schema()
        tool_def = {
            "name": "structured_output",
            "description": f"Return structured data as {output_format.__name__}",
            "input_schema": schema,
        }
        response: AnthropicMessage = await self._client.messages.create(
            **params,
            tools=[tool_def],
            tool_choice={"type": "tool", "name": "structured_output"},
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "structured_output":
                parsed = output_format.model_validate(block.input)
                return ChatInvokeCompletion(
                    completion=parsed,
                    stop_reason=response.stop_reason,
                    usage=_parse_usage(response.usage),
                )

        raise ValueError("No structured output returned from Anthropic API")

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool | dict] | None = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        params = _build_params(self._model, messages, self._merge_params(kwargs))

        anthropic_tools = _convert_tools(tools or [])
        if anthropic_tools:
            params["tools"] = anthropic_tools
            params["tool_choice"] = {
                "auto": {"type": "auto"},
                "required": {"type": "any"},
                "none": {"type": "none"},
            }[tool_choice]

        blocks: dict[int, dict[str, str]] = {}
        text_acc: list[str] = []
        stop_reason: str | None = None
        input_tokens = 0
        output_tokens = 0
        cache_read_tokens: int | None = None
        cache_creation_tokens: int | None = None
        saw_usage = False

        try:
            async with self._client.messages.stream(**params) as stream:
                async for event in stream:
                    if event.type == "message_start":
                        usage = event.message.usage
                        saw_usage = True
                        input_tokens = usage.input_tokens
                        cache_read_tokens = usage.cache_read_input_tokens
                        cache_creation_tokens = usage.cache_creation_input_tokens

                    elif event.type == "content_block_start":
                        content_block = event.content_block
                        if content_block.type == "tool_use":
                            blocks[event.index] = {
                                "type": "tool_use",
                                "id": content_block.id,
                                "name": content_block.name,
                                "json": "",
                            }
                        elif content_block.type == "text":
                            blocks[event.index] = {"type": "text"}

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            text_acc.append(delta.text)
                            yield StreamTextDelta(delta=delta.text)
                        elif delta.type == "input_json_delta":
                            block = blocks.get(event.index)
                            if block and block.get("type") == "tool_use":
                                block["json"] = (
                                    block.get("json", "") + delta.partial_json
                                )

                    elif event.type == "content_block_stop":
                        block = blocks.get(event.index)
                        if block and block.get("type") == "tool_use":
                            yield StreamToolCall(
                                tool_call=ToolCall(
                                    id=block["id"],
                                    function=Function(
                                        name=block["name"],
                                        arguments=block.get("json") or "{}",
                                    ),
                                )
                            )

                    elif event.type == "message_delta":
                        stop_reason = event.delta.stop_reason or stop_reason
                        saw_usage = True
                        output_tokens = event.usage.output_tokens
        except Exception as exc:
            mapped = _map_anthropic_error(exc)
            if mapped is not exc:
                raise mapped from exc
            raise

        usage_view: ChatInvokeUsage | None = None
        if saw_usage:
            usage_view = ChatInvokeUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                prompt_cached_tokens=cache_read_tokens,
                prompt_cache_creation_tokens=cache_creation_tokens,
            )

        yield StreamEnd(
            stop_reason=stop_reason,
            usage=usage_view,
            tool_calls=[
                ToolCall(
                    id=block["id"],
                    function=Function(
                        name=block["name"],
                        arguments=block.get("json") or "{}",
                    ),
                )
                for _, block in sorted(blocks.items())
                if block.get("type") == "tool_use"
            ],
            completion="".join(text_acc),
        )
