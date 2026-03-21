import os
import json
import httpx
from typing import Literal, Any, overload
from collections.abc import AsyncIterator

try:
    from anthropic import AsyncAnthropic
    from anthropic.types import Message as AnthropicMessage, Usage as AnthropicUsage
except ImportError:
    raise ImportError(
        "The 'anthropic' package is required for ChatAnthropic. "
        "Install it with: pip install py-llmify[anthropic]"
    )


from llmify.base import ChatModel
from llmify.messages import (
    Message,
    UserMessage,
    SystemMessage,
    AssistantMessage,
    ToolResultMessage,
    ContentPartTextParam,
    ContentPartImageParam,
    Function,
    ToolCall,
)
from llmify.tools import Tool
from llmify.views import ChatInvokeCompletion, ChatInvokeUsage


class ChatAnthropic(ChatModel):
    _client: AsyncAnthropic
    _model: str

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
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
        self._model = model

    @overload
    async def invoke[T](
        self, messages: list[Message], output_format: type[T], **kwargs: Any
    ) -> ChatInvokeCompletion[T]: ...

    @overload
    async def invoke(
        self, messages: list[Message], output_format: None = None, **kwargs: Any
    ) -> ChatInvokeCompletion[str]: ...

    async def invoke[T](
        self,
        messages: list[Message],
        output_format: type[T] | None = None,
        tools: list[Tool | dict] | None = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        **kwargs: Any,
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        params = self._build_params(messages, kwargs)

        if output_format is not None:
            return await self._invoke_with_structured_output(params, output_format)

        if tools:
            return await self._invoke_with_tools(params, tools, tool_choice)

        return await self._invoke_plain(params)

    def _build_params(
        self, messages: list[Message], method_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        merged = self._merge_params(method_kwargs)
        system_text, converted = self._convert_messages(messages)

        params: dict[str, Any] = {
            "model": self._model,
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
            if isinstance(stop, str):
                stop = [stop]
            params["stop_sequences"] = stop

        # Remove OpenAI-specific params that Anthropic doesn't support
        merged.pop("frequency_penalty", None)
        merged.pop("presence_penalty", None)
        merged.pop("seed", None)
        merged.pop("response_format", None)

        params.update(merged)
        return params

    async def _invoke_plain(self, params: dict[str, Any]) -> ChatInvokeCompletion[str]:
        response: AnthropicMessage = await self._client.messages.create(**params)
        return ChatInvokeCompletion(
            completion=self._extract_text(response),
            stop_reason=response.stop_reason,
            usage=self._parse_usage(response.usage),
        )

    async def _invoke_with_tools(
        self,
        params: dict[str, Any],
        tools: list[Tool | dict],
        tool_choice: Literal["auto", "required", "none"] = "auto",
    ) -> ChatInvokeCompletion[str]:
        anthropic_tools = [
            t if isinstance(t, dict) else self._convert_tool(t) for t in tools
        ]
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
            completion=self._extract_text(response),
            tool_calls=self._parse_tool_calls(response),
            stop_reason=response.stop_reason,
            usage=self._parse_usage(response.usage),
        )

    async def _invoke_with_structured_output[T](
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
                    usage=self._parse_usage(response.usage),
                )

        raise ValueError("No structured output returned from Anthropic API")

    def _extract_text(self, response: AnthropicMessage) -> str:
        parts = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "".join(parts)

    def _parse_usage(self, usage: AnthropicUsage) -> ChatInvokeUsage:
        return ChatInvokeUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
            prompt_cached_tokens=getattr(usage, "cache_read_input_tokens", None),
            prompt_cache_creation_tokens=getattr(
                usage, "cache_creation_input_tokens", None
            ),
        )

    def _parse_tool_calls(self, response: AnthropicMessage) -> list[ToolCall]:
        tool_calls = []
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        function=Function(
                            name=block.name,
                            arguments=json.dumps(block.input),
                        ),
                    )
                )
        return tool_calls

    def _convert_tool(self, tool: Tool) -> dict:
        openai_schema = tool.to_openai_schema()
        func = openai_schema.get("function", openai_schema)
        return {
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {}),
        }

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict]]:
        system_text: str | None = None
        converted: list[dict] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_text = msg.text
                continue

            if isinstance(msg, ToolResultMessage):
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )
                continue

            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                content: list[dict] = []
                if msg.text:
                    content.append({"type": "text", "text": msg.text})
                for tc in msg.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": json.loads(tc.function.arguments),
                        }
                    )
                converted.append({"role": "assistant", "content": content})
                continue

            if isinstance(msg, UserMessage) and isinstance(msg.content, list):
                content_parts: list[dict] = []
                for part in msg.content:
                    if isinstance(part, ContentPartTextParam):
                        content_parts.append({"type": "text", "text": part.text})
                    elif isinstance(part, ContentPartImageParam):
                        url = part.image_url.url
                        if url.startswith("data:"):
                            media_type, _, b64 = url.partition(";base64,")
                            media_type = media_type.replace("data:", "")
                            content_parts.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64,
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

            if isinstance(msg, UserMessage):
                converted.append({"role": "user", "content": msg.text})
            elif isinstance(msg, AssistantMessage):
                converted.append({"role": "assistant", "content": msg.text})

        return system_text, converted

    async def stream(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[str]:
        params = self._build_params(messages, kwargs)
        async with self._client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text
