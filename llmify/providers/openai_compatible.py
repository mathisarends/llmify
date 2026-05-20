from typing import Literal, Any, overload, TYPE_CHECKING
from collections.abc import AsyncIterator

from pydantic import BaseModel

try:
    from openai.types import CompletionUsage
    from openai import AsyncOpenAI, AsyncAzureOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
    )
except ImportError:
    if TYPE_CHECKING:
        raise

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
from llmify.views import (
    ChatInvokeCompletion,
    ChatInvokeUsage,
    StreamEnd,
    StreamEvent,
    StreamTextDelta,
    StreamToolCall,
)


class OpenAICompatible(ChatModel):
    _client: AsyncOpenAI | AsyncAzureOpenAI
    _model: str

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
        params = self._merge_params(kwargs)
        converted_messages = self._convert_messages(messages)

        if output_format is not None:
            return await self._invoke_with_structured_output(
                converted_messages, output_format, params
            )

        if tools:
            return await self._invoke_with_tools(
                converted_messages, tools, params, tool_choice
            )

        return await self._invoke_plain(converted_messages, params)

    async def _invoke_with_structured_output[T: BaseModel](
        self, messages: list[dict], output_format: type[T], params: dict[str, Any]
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
            usage=self._parse_usage(usage),
        )

    async def _invoke_with_tools(
        self,
        messages: list[dict],
        tools: list[Tool | dict],
        params: dict[str, Any],
        tool_choice: Literal["auto", "required", "none"] = "auto",
    ) -> ChatInvokeCompletion[str]:
        openai_tools = [
            t if isinstance(t, dict) else t.to_openai_schema() for t in tools
        ]
        response: ChatCompletion = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=openai_tools,
            tool_choice=tool_choice,
            **params,
        )
        choice = response.choices[0]
        return ChatInvokeCompletion(
            completion=choice.message.content or "",
            tool_calls=self._parse_tool_calls(choice.message.tool_calls),
            stop_reason=choice.finish_reason,
            usage=self._parse_usage(response.usage),
        )

    async def _invoke_plain(
        self, messages: list[dict], params: dict[str, Any]
    ) -> ChatInvokeCompletion[str]:
        response: ChatCompletion = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **params,
        )
        choice = response.choices[0]
        return ChatInvokeCompletion(
            completion=choice.message.content or "",
            stop_reason=choice.finish_reason,
            usage=self._parse_usage(response.usage),
        )

    def _parse_usage(self, usage: CompletionUsage | None) -> ChatInvokeUsage | None:
        if not usage:
            return None
        return ChatInvokeUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            prompt_cached_tokens=getattr(
                getattr(usage, "prompt_tokens_details", None), "cached_tokens", None
            ),
        )

    def _parse_tool_calls(
        self, raw_tool_calls: list[ChatCompletionMessageToolCall] | None
    ) -> list[ToolCall]:
        if not raw_tool_calls:
            return []
        return [
            ToolCall(
                id=tc.id,
                function=Function(
                    name=tc.function.name, arguments=tc.function.arguments
                ),
            )
            for tc in raw_tool_calls
        ]

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        return [self._convert_single_message(msg) for msg in messages]

    def _convert_single_message(self, msg: Message) -> dict:
        if isinstance(msg, ToolResultMessage):
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            }

        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            return {
                "role": "assistant",
                "content": msg.text or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }

        if isinstance(msg, UserMessage) and isinstance(msg.content, list):
            content = []
            for part in msg.content:
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
            return {"role": msg.role, "content": content}

        if isinstance(msg, (UserMessage, SystemMessage)):
            if isinstance(msg.content, list):
                return {
                    "role": msg.role,
                    "content": [{"type": "text", "text": p.text} for p in msg.content],
                }
            return {"role": msg.role, "content": msg.content}

        return {"role": msg.role, "content": msg.text}

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool | dict] | None = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        params = self._merge_params(kwargs)
        openai_tools = [
            t if isinstance(t, dict) else t.to_openai_schema() for t in tools or []
        ]

        raw_stream_options = params.pop("stream_options", None)
        stream_options: dict[str, Any] = (
            raw_stream_options.copy() if isinstance(raw_stream_options, dict) else {}
        )
        stream_options["include_usage"] = True

        request_args: dict[str, Any] = {
            "model": self._model,
            "messages": self._convert_messages(messages),
            "stream": True,
            "stream_options": stream_options,
            **params,
        }
        if openai_tools:
            request_args["tools"] = openai_tools
            request_args["tool_choice"] = tool_choice

        stream = await self._client.chat.completions.create(**request_args)

        buffers: dict[int, dict[str, Any]] = {}
        text_acc: list[str] = []
        stop_reason: str | None = None
        usage: ChatInvokeUsage | None = None

        chunk: ChatCompletionChunk
        async for chunk in stream:
            if chunk.usage is not None:
                usage = self._parse_usage(chunk.usage)

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
                        tool_call=ToolCall(
                            id=buf["id"],
                            function=Function(
                                name=buf["name"],
                                arguments=buf["arguments"],
                            ),
                        )
                    )

        yield StreamEnd(
            stop_reason=stop_reason,
            usage=usage,
            tool_calls=[
                ToolCall(
                    id=buffers[i]["id"],
                    function=Function(
                        name=buffers[i]["name"],
                        arguments=buffers[i]["arguments"],
                    ),
                )
                for i in sorted(buffers)
            ],
            completion="".join(text_acc),
        )
