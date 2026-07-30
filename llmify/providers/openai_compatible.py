from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from pydantic import BaseModel

try:
    from openai import AsyncAzureOpenAI, AsyncOpenAI
    from openai.types.chat import (
        ChatCompletion,
        ChatCompletionChunk,
        ChatCompletionMessageParam,
        ChatCompletionToolUnionParam,
    )
    from openai.types.chat.chat_completion_message import (
        ChatCompletionMessageToolCallUnion,
    )
    from openai.types.chat.chat_completion_message_function_tool_call import (
        ChatCompletionMessageFunctionToolCall,
    )
except ImportError:
    if TYPE_CHECKING:
        raise

from llmify.base import ChatModel
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
    parse_usage,
    reject_stream_parameter,
    tool_call,
    tool_schemas,
)
from llmify.retries import RetryCallback, retry_call, retry_stream
from llmify.tools import Tool
from llmify.views import (
    ChatInvokeCompletion,
    ChatInvokeUsage,
    StreamEnd,
    StreamEvent,
    StreamTextDelta,
    StreamToolCall,
)


def _convert_messages(messages: list[Message]) -> list[ChatCompletionMessageParam]:
    return [_convert_message(message) for message in messages]


def _convert_message(message: Message) -> ChatCompletionMessageParam:
    if isinstance(message, ToolResultMessage):
        return cast(
            ChatCompletionMessageParam,
            {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content,
            },
        )

    if isinstance(message, AssistantMessage) and message.tool_calls:
        return cast(
            ChatCompletionMessageParam,
            {
                "role": "assistant",
                "content": message.text or None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            },
        )

    if isinstance(message, UserMessage) and isinstance(message.content, list):
        content = []
        for part in message.content:
            if isinstance(part, ContentPartTextParam):
                content.append({"type": "text", "text": part.text})
            elif isinstance(part, ContentPartImageParam):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": part.image_url.url,
                            "detail": part.image_url.detail,
                        },
                    }
                )
        return cast(
            ChatCompletionMessageParam,
            {"role": message.role, "content": content},
        )

    if isinstance(message, SystemMessage):
        if isinstance(message.content, list):
            return cast(
                ChatCompletionMessageParam,
                {
                    "role": message.role,
                    "content": [
                        {"type": "text", "text": part.text} for part in message.content
                    ],
                },
            )
        return cast(
            ChatCompletionMessageParam,
            {"role": message.role, "content": message.content},
        )

    if isinstance(message, UserMessage):
        return cast(
            ChatCompletionMessageParam,
            {"role": message.role, "content": message.content},
        )

    return cast(
        ChatCompletionMessageParam,
        {"role": message.role, "content": message.text},
    )


def _parse_tool_calls(
    raw_tool_calls: list[ChatCompletionMessageToolCallUnion] | None,
) -> list[ToolCall]:
    if not raw_tool_calls:
        return []
    return [
        tool_call(
            call_id=raw_tool_call.id,
            name=raw_tool_call.function.name,
            arguments=raw_tool_call.function.arguments,
        )
        for raw_tool_call in raw_tool_calls
        if isinstance(raw_tool_call, ChatCompletionMessageFunctionToolCall)
    ]


class OpenAICompatible(ChatModel):
    _client: AsyncOpenAI | AsyncAzureOpenAI
    _model: str

    def __init__(self, *args: Any, **kwargs: Any):
        reject_stream_parameter(kwargs)
        super().__init__(*args, **kwargs)

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
        on_retry: RetryCallback | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        reject_stream_parameter(kwargs)

        async def invoke_once() -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
            params = self._merge_params(kwargs)
            converted_messages = _convert_messages(messages)

            if output_format is not None:
                return await self._invoke_with_structured_output(
                    converted_messages, output_format, params
                )

            if tools:
                return await self._invoke_with_tools(
                    converted_messages, tools, params, tool_choice
                )

            return await self._invoke_plain(converted_messages, params)

        return await retry_call(
            invoke_once,
            max_retries=self._default_max_retries,
            on_retry=on_retry if on_retry is not None else self._on_retry,
            map_error=map_openai_error,
        )

    async def _invoke_with_structured_output[T: BaseModel](
        self,
        messages: list[ChatCompletionMessageParam],
        output_format: type[T],
        params: dict[str, Any],
    ) -> ChatInvokeCompletion[T]:
        response = await self._client.beta.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=output_format,
            **params,
        )
        choice = response.choices[0]
        usage = response.usage
        return ChatInvokeCompletion(
            completion=choice.message.parsed,
            stop_reason=choice.finish_reason,
            usage=parse_usage(usage),
        )

    async def _invoke_with_tools(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[Tool | dict],
        params: dict[str, Any],
        tool_choice: Literal["auto", "required", "none"] = "auto",
    ) -> ChatInvokeCompletion[str]:
        openai_tools = cast(list[ChatCompletionToolUnionParam], tool_schemas(tools))
        create = cast(Any, self._client.chat.completions.create)
        response: ChatCompletion = await create(
            model=self._model,
            messages=messages,
            tools=openai_tools,
            tool_choice=tool_choice,
            **params,
        )
        choice = response.choices[0]
        return ChatInvokeCompletion(
            completion=choice.message.content or "",
            tool_calls=_parse_tool_calls(choice.message.tool_calls),
            stop_reason=choice.finish_reason,
            usage=parse_usage(response.usage),
        )

    async def _invoke_plain(
        self,
        messages: list[ChatCompletionMessageParam],
        params: dict[str, Any],
    ) -> ChatInvokeCompletion[str]:
        create = cast(Any, self._client.chat.completions.create)
        response: ChatCompletion = await create(
            model=self._model,
            messages=messages,
            **params,
        )
        choice = response.choices[0]
        return ChatInvokeCompletion(
            completion=choice.message.content or "",
            stop_reason=choice.finish_reason,
            usage=parse_usage(response.usage),
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool | dict] | None = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        on_retry: RetryCallback | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        reject_stream_parameter(kwargs)
        params = self._merge_params(kwargs)
        openai_tools = tool_schemas(tools or [])

        raw_stream_options = params.pop("stream_options", None)
        stream_options: dict[str, Any] = (
            raw_stream_options.copy() if isinstance(raw_stream_options, dict) else {}
        )
        stream_options["include_usage"] = True

        request_args: dict[str, Any] = {
            "model": self._model,
            "messages": _convert_messages(messages),
            **params,
            "stream": True,
            "stream_options": stream_options,
        }
        if openai_tools:
            request_args["tools"] = openai_tools
            request_args["tool_choice"] = tool_choice

        async for event in retry_stream(
            lambda: self._stream_once(request_args),
            max_retries=self._default_max_retries,
            on_retry=on_retry if on_retry is not None else self._on_retry,
            map_error=map_openai_error,
        ):
            yield event

    async def _stream_once(
        self, request_args: dict[str, Any]
    ) -> AsyncIterator[StreamEvent]:
        stream = await self._client.chat.completions.create(**request_args)
        buffers: dict[int, dict[str, Any]] = {}
        text_acc: list[str] = []
        stop_reason: str | None = None
        usage: ChatInvokeUsage | None = None

        chunk: ChatCompletionChunk
        async for chunk in stream:
            if chunk.usage is not None:
                usage = parse_usage(chunk.usage)

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                text_acc.append(delta.content)
                yield StreamTextDelta(delta=delta.content)

            for tc_delta in delta.tool_calls or []:
                buf = buffers.setdefault(
                    tc_delta.index,
                    {"id": "", "name": "", "arguments": "", "emitted": False},
                )
                if tc_delta.id:
                    buf["id"] = tc_delta.id

                if tc_delta.function:
                    if tc_delta.function.name:
                        buf["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        buf["arguments"] += tc_delta.function.arguments

            if choice.finish_reason:
                stop_reason = choice.finish_reason
                for idx in sorted(buffers):
                    buf = buffers[idx]
                    if buf["emitted"]:
                        continue

                    buf["emitted"] = True
                    yield StreamToolCall(
                        tool_call=tool_call(
                            call_id=buf["id"],
                            name=buf["name"],
                            arguments=buf["arguments"],
                        )
                    )

        yield StreamEnd(
            stop_reason=stop_reason,
            usage=usage,
            tool_calls=[
                tool_call(
                    call_id=buffers[i]["id"],
                    name=buffers[i]["name"],
                    arguments=buffers[i]["arguments"],
                )
                for i in sorted(buffers)
            ],
            completion="".join(text_acc),
        )
