import json
import os
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any, Literal, cast, overload

import httpx
from pydantic import BaseModel

try:
    from google import genai
    from google.genai import errors as google_errors
    from google.genai import types as google_types
    from google.genai.client import AsyncClient
except ImportError:
    raise ImportError(
        "The 'google-genai' package is required for ChatGoogle. "
        "Install it with: pip install py-llmify[google]"
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


def _map_google_error(exc: Exception) -> Exception:
    if isinstance(exc, httpx.TransportError):
        return RetryableError(str(exc))
    if not isinstance(exc, google_errors.APIError):
        return exc

    status_code = exc.code
    message = exc.message or str(exc)
    message_lower = message.lower()

    if status_code == 429:
        retry_after: float | None = None
        response = getattr(exc, "response", None)
        raw = response.headers.get("retry-after") if response is not None else None
        if raw:
            try:
                retry_after = float(raw)
            except ValueError:
                pass
        return RateLimitError(str(exc), retry_after=retry_after)
    if status_code == 402:
        return OutOfCreditsError(str(exc))
    if status_code in (401, 403):
        return AuthenticationError(str(exc))
    if status_code == 400 and (
        "context" in message_lower
        or (
            "token" in message_lower
            and any(kw in message_lower for kw in ("exceed", "maximum", "limit"))
        )
    ):
        return ContextLengthExceededError(str(exc))
    if status_code is not None and (status_code in {408, 409} or status_code >= 500):
        return RetryableError(str(exc), status_code=status_code)
    return exc


class GoogleModel(StrEnum):
    GEMINI_3_6_FLASH = "gemini-3.6-flash"
    GEMINI_3_5_FLASH = "gemini-3.5-flash"
    GEMINI_3_5_FLASH_LITE = "gemini-3.5-flash-lite"
    GEMINI_3_1_FLASH_LITE = "gemini-3.1-flash-lite"

    GEMINI_3_1_PRO_PREVIEW = "gemini-3.1-pro-preview"
    GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview"


def _build_config(
    merged: dict[str, Any],
    *,
    system_instruction: str | None = None,
    tools: list[Tool | dict[str, Any]] | None = None,
    tool_choice: Literal["auto", "required", "none"] = "auto",
    output_format: type[BaseModel] | None = None,
) -> google_types.GenerateContentConfig | None:
    config: dict[str, Any] = {}

    if "max_tokens" in merged:
        config["max_output_tokens"] = merged.pop("max_tokens")
    if "temperature" in merged:
        config["temperature"] = merged.pop("temperature")
    if "top_p" in merged:
        config["top_p"] = merged.pop("top_p")
    if "frequency_penalty" in merged:
        config["frequency_penalty"] = merged.pop("frequency_penalty")
    if "presence_penalty" in merged:
        config["presence_penalty"] = merged.pop("presence_penalty")
    if "stop" in merged:
        stop = merged.pop("stop")
        config["stop_sequences"] = [stop] if isinstance(stop, str) else stop
    if "seed" in merged:
        config["seed"] = merged.pop("seed")
    if "response_format" in merged:
        response_format = merged.pop("response_format")
        if isinstance(response_format, dict):
            config.update(response_format)

    if system_instruction:
        config["system_instruction"] = system_instruction

    if output_format is not None:
        config["response_mime_type"] = "application/json"
        config["response_json_schema"] = output_format.model_json_schema()

    google_tools = [_convert_tool(tool) for tool in tools or []]
    if google_tools and tool_choice != "none":
        config["tools"] = [{"function_declarations": google_tools}]
        config["automatic_function_calling"] = {"disable": True}
        mode = "ANY" if tool_choice == "required" else "AUTO"
        config["tool_config"] = {"function_calling_config": {"mode": mode}}

    config.update(merged)
    if not config:
        return None
    return google_types.GenerateContentConfig(**config)


def _convert_messages(
    messages: list[Message],
) -> tuple[list[dict[str, Any]], str | None]:
    contents: list[dict[str, Any]] = []
    system_instruction: str | None = None
    tool_names_by_id: dict[str, str] = {}

    for message in messages:
        if isinstance(message, SystemMessage):
            system_instruction = message.text
            continue

        if isinstance(message, ToolResultMessage):
            name = tool_names_by_id.get(message.tool_call_id, message.tool_call_id)
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "id": message.tool_call_id,
                                "name": name,
                                "response": {"result": message.content},
                            }
                        }
                    ],
                }
            )
            continue

        if isinstance(message, AssistantMessage) and message.tool_calls:
            parts: list[dict[str, Any]] = []
            if message.text:
                parts.append({"text": message.text})
            for tool_call in message.tool_calls:
                tool_names_by_id[tool_call.id] = tool_call.function.name
                part: dict[str, Any] = {
                    "function_call": {
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "args": json.loads(tool_call.function.arguments or "{}"),
                    }
                }
                google_metadata = tool_call.provider_metadata.get("google")
                if isinstance(google_metadata, dict):
                    thought_signature = google_metadata.get("thought_signature")
                    if isinstance(thought_signature, bytes):
                        part["thought_signature"] = thought_signature
                parts.append(part)
            contents.append({"role": "model", "parts": parts})
            continue

        if isinstance(message, UserMessage):
            contents.append({"role": "user", "parts": _convert_user_parts(message)})
        elif isinstance(message, AssistantMessage):
            contents.append({"role": "model", "parts": [{"text": message.text}]})

    return contents, system_instruction


def _convert_user_parts(message: UserMessage) -> list[dict[str, Any]]:
    if isinstance(message.content, str):
        return [{"text": message.content}]

    parts: list[dict[str, Any]] = []
    for part in message.content:
        if isinstance(part, ContentPartTextParam):
            parts.append({"text": part.text})
        elif isinstance(part, ContentPartImageParam):
            url = part.image_url.url
            if url.startswith("data:"):
                media_type, _, data = url.partition(";base64,")
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": media_type.removeprefix("data:"),
                            "data": data,
                        }
                    }
                )
            else:
                parts.append(
                    {
                        "file_data": {
                            "mime_type": part.image_url.media_type,
                            "file_uri": url,
                        }
                    }
                )
    return parts


def _convert_tool(tool: Tool | dict[str, Any]) -> dict[str, Any]:
    openai_schema = tool if isinstance(tool, dict) else tool.to_openai_schema()
    function = openai_schema.get("function", openai_schema)
    if not isinstance(function, dict):
        raise TypeError("Tool schema must contain a function object")
    function = cast(dict[str, Any], function)
    if "parameters_json_schema" in function:
        return function
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "parameters_json_schema": function.get("parameters", {}),
    }


def _parse_tool_calls(
    response: google_types.GenerateContentResponse,
) -> list[ToolCall]:
    if (
        not response.candidates
        or response.candidates[0].content is None
        or not response.candidates[0].content.parts
    ):
        return []

    tool_calls: list[ToolCall] = []
    for part in response.candidates[0].content.parts:
        if part.function_call is None:
            continue

        tool_call = _parse_function_call(
            part.function_call,
            index=len(tool_calls),
        )
        if part.thought_signature is not None:
            tool_call.provider_metadata["google"] = {
                "thought_signature": part.thought_signature
            }
        tool_calls.append(tool_call)
    return tool_calls


def _parse_text(response: google_types.GenerateContentResponse) -> str:
    if (
        not response.candidates
        or response.candidates[0].content is None
        or not response.candidates[0].content.parts
    ):
        return ""

    return "".join(
        part.text
        for part in response.candidates[0].content.parts
        if isinstance(part.text, str) and part.thought is not True
    )


def _parse_function_call(
    function_call: google_types.FunctionCall, *, index: int
) -> ToolCall:
    if function_call.name is None:
        raise ValueError("Google returned a function call without a name")

    return ToolCall(
        id=function_call.id or f"call_{index}_{function_call.name}",
        function=Function(
            name=function_call.name,
            arguments=json.dumps(function_call.args or {}),
        ),
    )


def _parse_usage(
    usage: google_types.GenerateContentResponseUsageMetadata | None,
) -> ChatInvokeUsage | None:
    if usage is None:
        return None

    prompt_tokens = usage.prompt_token_count or 0
    completion_tokens = usage.candidates_token_count or 0
    total_tokens = usage.total_token_count
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens

    return ChatInvokeUsage(
        prompt_tokens=prompt_tokens,
        prompt_cached_tokens=usage.cached_content_token_count,
        prompt_image_tokens=_image_token_count(usage),
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _image_token_count(
    usage: google_types.GenerateContentResponseUsageMetadata,
) -> int | None:
    total = sum(
        detail.token_count or 0
        for detail in usage.prompt_tokens_details or []
        if detail.modality == google_types.MediaModality.IMAGE
    )
    return total or None


def _stop_reason(
    response: google_types.GenerateContentResponse,
) -> str | None:
    if not response.candidates:
        return None
    finish_reason = response.candidates[0].finish_reason
    return finish_reason.value if finish_reason is not None else None


class ChatGoogle(ChatModel):
    _client: AsyncClient
    _model: str

    def __init__(
        self,
        model: str | GoogleModel = "gemini-3.5-flash",
        api_key: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | list[str] | None = None,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
        timeout: float | httpx.Timeout | None = 60.0,
        max_retries: int = 2,
        on_retry: RetryCallback | None = None,
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
            on_retry=on_retry,
            **kwargs,
        )
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")

        self._client = genai.Client(
            api_key=api_key,
            http_options=google_types.HttpOptions(
                retry_options=google_types.HttpRetryOptions(attempts=1)
            ),
        ).aio

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
        tools: list[Tool | dict[str, Any]] | None = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        on_retry: RetryCallback | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        async def invoke_once() -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
            contents, system_instruction = _convert_messages(messages)
            config = _build_config(
                self._merge_params(kwargs),
                system_instruction=system_instruction,
                tools=tools,
                tool_choice=tool_choice,
                output_format=output_format,
            )
            response = await self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config or None,
            )

            if output_format is not None:
                return ChatInvokeCompletion(
                    completion=output_format.model_validate_json(
                        _parse_text(response) or "{}"
                    ),
                    stop_reason=_stop_reason(response),
                    usage=_parse_usage(response.usage_metadata),
                )

            return ChatInvokeCompletion(
                completion=_parse_text(response),
                tool_calls=_parse_tool_calls(response),
                stop_reason=_stop_reason(response),
                usage=_parse_usage(response.usage_metadata),
            )

        return await retry_call(
            invoke_once,
            max_retries=self._default_max_retries,
            on_retry=on_retry if on_retry is not None else self._on_retry,
            map_error=_map_google_error,
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool | dict[str, Any]] | None = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        on_retry: RetryCallback | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        contents, system_instruction = _convert_messages(messages)
        config = _build_config(
            self._merge_params(kwargs),
            system_instruction=system_instruction,
            tools=tools,
            tool_choice=tool_choice,
        )

        async for event in retry_stream(
            lambda: self._stream_once(contents, config),
            max_retries=self._default_max_retries,
            on_retry=on_retry if on_retry is not None else self._on_retry,
            map_error=_map_google_error,
        ):
            yield event

    async def _stream_once(
        self,
        contents: list[dict[str, Any]],
        config: google_types.GenerateContentConfig | None,
    ) -> AsyncIterator[StreamEvent]:
        stream = await self._client.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config or None,
        )

        text_acc: list[str] = []
        tool_calls: list[ToolCall] = []
        stop_reason: str | None = None
        usage: ChatInvokeUsage | None = None

        async for chunk in stream:
            chunk_text = _parse_text(chunk)
            if chunk_text:
                text_acc.append(chunk_text)
                yield StreamTextDelta(delta=chunk_text)

            for tool_call in _parse_tool_calls(chunk):
                tool_calls.append(tool_call)
                yield StreamToolCall(tool_call=tool_call)

            stop_reason = _stop_reason(chunk) or stop_reason
            chunk_usage = _parse_usage(chunk.usage_metadata)
            if chunk_usage is not None:
                usage = chunk_usage

        yield StreamEnd(
            stop_reason=stop_reason,
            usage=usage,
            tool_calls=tool_calls,
            completion="".join(text_acc),
        )
