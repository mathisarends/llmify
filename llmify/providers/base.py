from abc import ABC, abstractmethod
import json
from typing import Literal, Self, TypeVar, Any, Generic
from collections.abc import AsyncIterator

from pydantic import BaseModel
import httpx

from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from llmify.messages import (
    Message,
    ImageMessage,
    ToolResultMessage,
    AssistantToolCallMessage,
    ModelResponse,
    ToolCall,
)
from llmify.tools import Tool

T = TypeVar("T", bound=BaseModel)


class BaseChatModel(ABC, Generic[T]):
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
        _response_model: type[T] | None = None,
        **kwargs: Any,
    ):
        self._default_max_tokens = max_tokens
        self._default_temperature = temperature
        self._default_top_p = top_p
        self._default_frequency_penalty = frequency_penalty
        self._default_presence_penalty = presence_penalty
        self._default_stop = stop
        self._default_seed = seed
        self._default_response_format = response_format
        self._default_timeout = timeout
        self._default_max_retries = max_retries
        self._default_kwargs = kwargs
        self._response_model = _response_model

    def _merge_params(self, method_kwargs: dict[str, Any]) -> dict[str, Any]:
        defaults = {
            "max_tokens": self._default_max_tokens,
            "temperature": self._default_temperature,
            "top_p": self._default_top_p,
            "frequency_penalty": self._default_frequency_penalty,
            "presence_penalty": self._default_presence_penalty,
            "stop": self._default_stop,
            "seed": self._default_seed,
            "response_format": self._default_response_format,
        }

        params = {**self._default_kwargs}

        for key, default in defaults.items():
            value = method_kwargs.get(key, default)
            if value is not None:
                params[key] = value

        for key, value in method_kwargs.items():
            if key not in defaults and value is not None:
                params[key] = value

        return params

    def with_structured_output(self, response_model: type[T]) -> Self:
        self._response_model = response_model
        return self

    @abstractmethod
    async def invoke(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str | T | ModelResponse:
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        pass


class BaseOpenAICompatible(BaseChatModel[T]):
    _client: AsyncOpenAI | AsyncAzureOpenAI
    _model: str

    def with_structured_output(
        self, response_model: type[T]
    ) -> "BaseOpenAICompatible[T]":
        instance = self.__class__(
            max_tokens=self._default_max_tokens,
            temperature=self._default_temperature,
            top_p=self._default_top_p,
            frequency_penalty=self._default_frequency_penalty,
            presence_penalty=self._default_presence_penalty,
            stop=self._default_stop,
            seed=self._default_seed,
            response_format=self._default_response_format,
            timeout=self._default_timeout,
            max_retries=self._default_max_retries,
            _response_model=response_model,
            **self._default_kwargs,
        )
        instance._client = self._client
        instance._model = self._model
        return instance

    async def invoke(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[Tool | dict] | None = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        **kwargs: Any,
    ) -> str | T | ModelResponse:
        params = self._merge_params(
            {"max_tokens": max_tokens, "temperature": temperature, **kwargs}
        )
        converted_messages = self._convert_messages(messages)

        if self._response_model is not None:
            return await self._invoke_with_structured_output(converted_messages, params)

        if tools:
            return await self._invoke_with_tools(
                converted_messages, tools, params, tool_choice
            )

        return await self._invoke_plain(converted_messages, params)

    async def _invoke_with_structured_output(
        self, messages: list[dict], params: dict[str, Any]
    ) -> T:
        response = await self._client.beta.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=self._response_model,
            **params,
        )
        return response.choices[0].message.parsed

    async def _invoke_with_tools(
        self,
        messages: list[dict],
        tools: list[Tool | dict],
        params: dict[str, Any],
        tool_choice: Literal["auto", "required", "none"] = "auto",
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
        tool_calls = self._parse_tool_calls(choice.message.tool_calls, tool_registry)

        return ModelResponse(
            content=choice.message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
        )

    def _parse_tool_calls(
        self,
        raw_tool_calls: list[ChatCompletionMessageToolCall] | None,
        tool_registry: dict[str, Tool],
    ) -> list[ToolCall]:
        if not raw_tool_calls:
            return []

        tool_calls = []
        for tc in raw_tool_calls:
            name = tc.function.name

            if name in tool_registry:
                parsed_args = tool_registry[name].parse_arguments(tc.function.arguments)
            else:
                parsed_args = json.loads(tc.function.arguments)

            tool_calls.append(ToolCall(id=tc.id, name=name, tool=parsed_args))

        return tool_calls

    async def _invoke_plain(self, messages: list[dict], params: dict[str, Any]) -> str:
        response: ChatCompletion = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **params,
        )
        return response.choices[0].message.content or ""

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        return [self._convert_single_message(msg) for msg in messages]

    def _convert_single_message(self, msg: Message) -> dict:
        if isinstance(msg, ToolResultMessage):
            return self._convert_tool_result(msg)

        if isinstance(msg, AssistantToolCallMessage):
            return self._convert_assistant_tool_call(msg)

        if isinstance(msg, ImageMessage):
            return self._convert_image_message(msg)

        return {"role": msg.role, "content": msg.content}

    def _convert_tool_result(self, msg: ToolResultMessage) -> dict:
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": msg.content,
        }

    def _convert_assistant_tool_call(self, msg: AssistantToolCallMessage) -> dict:
        tool_calls_list = []
        for tc in msg.tool_calls:
            # Handle both Pydantic models and other tool result types
            if isinstance(tc.tool, BaseModel):
                arguments_json = tc.tool.model_dump_json()
            elif isinstance(tc.tool, dict):
                import json

                arguments_json = json.dumps(tc.tool)
            else:
                import json

                arguments_json = json.dumps(tc.tool)

            tool_calls_list.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": arguments_json,
                    },
                }
            )

        return {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": tool_calls_list,
        }

    def _convert_image_message(self, msg: ImageMessage) -> dict:
        content = []

        if msg.content:
            content.append({"type": "text", "text": msg.content})

        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{msg.media_type};base64,{msg.base64_data}",
                    "detail": msg.detail,
                },
            }
        )

        return {"role": msg.role, "content": content}

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
