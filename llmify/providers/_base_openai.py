from abc import ABC
from collections.abc import AsyncIterator
from typing import Any, Literal, TypeVar, overload
import httpx
from pydantic import BaseModel
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from llmify.messages import (
    AssistantMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    Message,
    ModelResponse,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from llmify.tools import Tool
from llmify.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar("T", bound=BaseModel)

_KNOWN_PARAM_KEYS = {
    "max_tokens",
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "seed",
    "response_format",
}


class BaseOpenAICompatible(ABC):
    _client: AsyncOpenAI | AsyncAzureOpenAI
    _model: str

    def __init__(
        self,
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
        **kwargs: Any,
    ) -> None:
        self._defaults = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "seed": seed,
            "response_format": response_format,
        }
        self._extra_defaults = kwargs

    @overload
    async def invoke(
        self,
        messages: list[Message],
        *,
        tools: list[Tool | dict],
        tool_choice: Literal["auto", "required", "none"] = ...,
        max_tokens: int | None = ...,
        temperature: float | None = ...,
        **kwargs: Any,
    ) -> ModelResponse: ...

    @overload
    async def invoke(
        self,
        messages: list[Message],
        *,
        response_model: type[T],
        max_tokens: int | None = ...,
        temperature: float | None = ...,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[T]: ...

    @overload
    async def invoke(
        self,
        messages: list[Message],
        max_tokens: int | None = ...,
        temperature: float | None = ...,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[str]: ...

    async def invoke(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[Tool | dict] | None = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        response_model: type[Any] | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[str] | ChatInvokeCompletion[Any] | ModelResponse:
        params = self._merge_params(
            {"max_tokens": max_tokens, "temperature": temperature, **kwargs}
        )
        converted = self._convert_messages(messages)

        if response_model is not None:
            return await self._invoke_structured(converted, params, response_model)
        if tools:
            return await self._invoke_with_tools(converted, tools, params, tool_choice)
        return await self._invoke_plain(converted, params)

    async def stream(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        params = self._merge_params(
            {"max_tokens": max_tokens, "temperature": temperature, **kwargs}
        )
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=self._convert_messages(messages),
            stream=True,
            **params,
        )
        chunk: ChatCompletionChunk
        async for chunk in stream:
            if content := chunk.choices[0].delta.content:
                yield content

    async def _invoke_plain(
        self, messages: list[dict], params: dict[str, Any]
    ) -> ChatInvokeCompletion[str]:
        response: ChatCompletion = await self._client.chat.completions.create(
            model=self._model, messages=messages, **params
        )
        choice = response.choices[0]
        return ChatInvokeCompletion(
            completion=choice.message.content or "",
            usage=self._extract_usage(response),
            stop_reason=choice.finish_reason,
        )

    async def _invoke_structured(
        self, messages: list[dict], params: dict[str, Any], response_model: type[T]
    ) -> ChatInvokeCompletion[T]:
        response = await self._client.beta.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=response_model,
            **params,
        )
        choice = response.choices[0]
        return ChatInvokeCompletion(
            completion=choice.message.parsed,
            usage=self._extract_usage(response),
            stop_reason=choice.finish_reason,
        )

    async def _invoke_with_tools(
        self,
        messages: list[dict],
        tools: list[Tool | dict],
        params: dict[str, Any],
        tool_choice: Literal["auto", "required", "none"],
    ) -> ModelResponse:
        openai_tools = [
            t if isinstance(t, dict) else t.to_openai_schema() for t in tools
        ]
        tool_registry = {t.name: t for t in tools if not isinstance(t, dict)}
        response: ChatCompletion = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=openai_tools,
            tool_choice=tool_choice,
            **params,
        )
        choice = response.choices[0]
        return ModelResponse(
            content=choice.message.content,
            tool_calls=self._parse_tool_calls(choice.message.tool_calls, tool_registry),
            finish_reason=choice.finish_reason,
        )

    def _parse_tool_calls(
        self, raw: list[ChatCompletionMessageToolCall] | None, registry: dict[str, Tool]
    ) -> list[ToolCall]:
        if not raw:
            return []
        result = []
        for tc in raw:
            if tc.function.name not in registry:
                raise ValueError(f"Unknown tool: {tc.function.name}")
            result.append(
                ToolCall(
                    id=tc.id,
                    function={
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                    tool=registry[tc.function.name].parse_arguments(
                        tc.function.arguments
                    ),
                )
            )
        return result

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        return [self._convert_message(msg) for msg in messages]

    def _convert_message(self, msg: Message) -> dict:
        if isinstance(msg, ToolResultMessage):
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            }
        if isinstance(msg, AssistantMessage) and msg.has_tool_calls:
            return {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        if isinstance(msg, AssistantMessage):
            return {"role": "assistant", "content": msg.content}
        if isinstance(msg, UserMessage) and isinstance(msg.content, list):
            return {"role": "user", "content": self._convert_parts(msg.content)}
        if isinstance(msg, UserMessage):
            return {"role": "user", "content": msg.content}
        return {
            "role": msg.role,
            "content": msg.text if isinstance(msg, SystemMessage) else msg.content,
        }  # type: ignore[union-attr]

    def _convert_parts(
        self, parts: list[ContentPartTextParam | ContentPartImageParam]
    ) -> list[dict]:
        return [
            {"type": "text", "text": p.text}
            if isinstance(p, ContentPartTextParam)
            else {
                "type": "image_url",
                "image_url": {"url": p.image_url.url, "detail": p.image_url.detail},
            }
            for p in parts
        ]

    def _merge_params(self, overrides: dict[str, Any]) -> dict[str, Any]:
        params = {**self._extra_defaults}
        for key, default in self._defaults.items():
            value = overrides.get(key, default)
            if value is not None:
                params[key] = value
        for key, value in overrides.items():
            if key not in _KNOWN_PARAM_KEYS and value is not None:
                params[key] = value
        return params

    def _extract_usage(self, response: ChatCompletion) -> ChatInvokeUsage | None:
        if not response.usage:
            return None
        details = getattr(response.usage, "prompt_tokens_details", None)
        return ChatInvokeUsage(
            prompt_tokens=response.usage.prompt_tokens,
            prompt_cached_tokens=getattr(details, "cached_tokens", None)
            if details
            else None,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
