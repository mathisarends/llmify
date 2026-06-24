import json
import os
from collections.abc import AsyncIterator
from typing import Any, Literal, overload

import httpx
from pydantic import BaseModel, Field

try:
    from google import genai
    from google.genai import errors as google_errors
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
    if not isinstance(exc, google_errors.APIError):
        return exc

    status_code = getattr(exc, "code", None)
    message = getattr(exc, "message", str(exc))
    message_lower = message.lower()

    if status_code == 429:
        return RateLimitError(str(exc))
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
    if status_code is not None and status_code >= 500:
        return RetryableError(str(exc), status_code=status_code)
    return exc


class ChatGoogle(ChatModel):
    _client: Any
    _model: str

    def __init__(
        self,
        model: str = "gemini-3.5-flash",
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
            api_key = os.getenv("GEMINI_API_KEY")

        self._client = genai.Client(api_key=api_key).aio

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
        **kwargs: Any,
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
        try:
            contents, system_instruction = self._convert_messages(messages)
            config = self._build_config(
                kwargs,
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
                    completion=output_format.model_validate_json(response.text or "{}"),
                    stop_reason=self._stop_reason(response),
                    usage=self._parse_usage(getattr(response, "usage_metadata", None)),
                )

            return ChatInvokeCompletion(
                completion=response.text or "",
                tool_calls=self._parse_tool_calls(response),
                stop_reason=self._stop_reason(response),
                usage=self._parse_usage(getattr(response, "usage_metadata", None)),
            )
        except Exception as exc:
            mapped = _map_google_error(exc)
            if mapped is not exc:
                raise mapped from exc
            raise

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool | dict[str, Any]] | None = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        contents, system_instruction = self._convert_messages(messages)
        config = self._build_config(
            kwargs,
            system_instruction=system_instruction,
            tools=tools,
            tool_choice=tool_choice,
        )

        try:
            stream = await self._client.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=config or None,
            )
        except Exception as exc:
            mapped = _map_google_error(exc)
            if mapped is not exc:
                raise mapped from exc
            raise

        text_acc: list[str] = []
        tool_calls: list[ToolCall] = []
        stop_reason: str | None = None
        usage: ChatInvokeUsage | None = None

        try:
            async for chunk in stream:
                if getattr(chunk, "text", None):
                    text_acc.append(chunk.text)
                    yield StreamTextDelta(delta=chunk.text)

                for tool_call in self._parse_tool_calls(chunk):
                    tool_calls.append(tool_call)
                    yield StreamToolCall(tool_call=tool_call)

                stop_reason = self._stop_reason(chunk) or stop_reason
                chunk_usage = self._parse_usage(getattr(chunk, "usage_metadata", None))
                if chunk_usage is not None:
                    usage = chunk_usage
        except Exception as exc:
            mapped = _map_google_error(exc)
            if mapped is not exc:
                raise mapped from exc
            raise

        yield StreamEnd(
            stop_reason=stop_reason,
            usage=usage,
            tool_calls=tool_calls,
            completion="".join(text_acc),
        )

    def _build_config(
        self,
        method_kwargs: dict[str, Any],
        *,
        system_instruction: str | None = None,
        tools: list[Tool | dict[str, Any]] | None = None,
        tool_choice: Literal["auto", "required", "none"] = "auto",
        output_format: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        merged = self._merge_params(method_kwargs)
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

        google_tools = [self._convert_tool(t) for t in tools or []]
        if google_tools and tool_choice != "none":
            config["tools"] = [{"function_declarations": google_tools}]
            config["automatic_function_calling"] = {"disable": True}
            mode = "ANY" if tool_choice == "required" else "AUTO"
            config["tool_config"] = {"function_calling_config": {"mode": mode}}

        config.update(merged)
        return config

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[list[dict[str, Any]], str | None]:
        contents: list[dict[str, Any]] = []
        system_instruction: str | None = None
        tool_names_by_id: dict[str, str] = {}

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_instruction = msg.text
                continue

            if isinstance(msg, ToolResultMessage):
                name = tool_names_by_id.get(msg.tool_call_id, msg.tool_call_id)
                contents.append(
                    {
                        "role": "tool",
                        "parts": [
                            {
                                "function_response": {
                                    "name": name,
                                    "response": {"result": msg.content},
                                }
                            }
                        ],
                    }
                )
                continue

            if isinstance(msg, AssistantMessage) and msg.tool_calls:
                parts: list[dict[str, Any]] = []
                if msg.text:
                    parts.append({"text": msg.text})
                for tc in msg.tool_calls:
                    tool_names_by_id[tc.id] = tc.function.name
                    parts.append(
                        {
                            "function_call": {
                                "id": tc.id,
                                "name": tc.function.name,
                                "args": json.loads(tc.function.arguments or "{}"),
                            }
                        }
                    )
                contents.append({"role": "model", "parts": parts})
                continue

            if isinstance(msg, UserMessage):
                contents.append(
                    {"role": "user", "parts": self._convert_user_parts(msg)}
                )
            elif isinstance(msg, AssistantMessage):
                contents.append({"role": "model", "parts": [{"text": msg.text}]})

        return contents, system_instruction

    def _convert_user_parts(self, msg: UserMessage) -> list[dict[str, Any]]:
        if isinstance(msg.content, str):
            return [{"text": msg.content}]

        parts: list[dict[str, Any]] = []
        for part in msg.content:
            if isinstance(part, ContentPartTextParam):
                parts.append({"text": part.text})
            elif isinstance(part, ContentPartImageParam):
                url = part.image_url.url
                if url.startswith("data:"):
                    media_type, _, data = url.partition(";base64,")
                    parts.append(
                        {
                            "inline_data": {
                                "mime_type": media_type.replace("data:", ""),
                                "data": data,
                            }
                        }
                    )
                else:
                    parts.append(
                        {
                            "file_data": {
                                "mime_type": part.image_url.media_type,
                                "file_uri": part.image_url.url,
                            }
                        }
                    )
        return parts

    def _convert_tool(self, tool: Tool | dict[str, Any]) -> dict[str, Any]:
        openai_schema = tool if isinstance(tool, dict) else tool.to_openai_schema()
        func = openai_schema.get("function", openai_schema)
        if "parameters_json_schema" in func:
            return func
        return {
            "name": func["name"],
            "description": func.get("description", ""),
            "parameters_json_schema": func.get("parameters", {}),
        }

    def _parse_tool_calls(self, response: object) -> list[ToolCall]:
        raw_calls = getattr(response, "function_calls", None) or []
        tool_calls = []
        for index, raw_call in enumerate(raw_calls):
            parsed = _GoogleFunctionCall.from_raw(raw_call, index=index)
            tool_calls.append(
                ToolCall(
                    id=parsed.id,
                    function=Function(
                        name=parsed.name,
                        arguments=json.dumps(parsed.args),
                    ),
                )
            )
        return tool_calls

    def _parse_usage(self, usage: object | None) -> ChatInvokeUsage | None:
        if usage is None:
            return None

        parsed = _GoogleUsage.from_raw(usage)

        return ChatInvokeUsage(
            prompt_tokens=parsed.prompt_tokens,
            prompt_cached_tokens=parsed.prompt_cached_tokens,
            prompt_image_tokens=parsed.prompt_image_tokens,
            completion_tokens=parsed.completion_tokens,
            total_tokens=parsed.total_tokens,
        )

    def _image_token_count(self, usage: object) -> int | None:
        return _google_image_token_count(usage)

    def _stop_reason(self, response: object) -> str | None:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        finish_reason = getattr(candidates[0], "finish_reason", None)
        return str(finish_reason) if finish_reason is not None else None


class _GoogleFunctionCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw_call: object, *, index: int) -> "_GoogleFunctionCall":
        function_call = getattr(raw_call, "function_call", raw_call)
        name = getattr(raw_call, "name", None) or getattr(function_call, "name")
        args = getattr(raw_call, "args", None) or getattr(function_call, "args", None)
        call_id = getattr(raw_call, "id", None) or f"call_{index}_{name}"
        return cls(id=call_id, name=name, args=args or {})


class _GoogleUsage(BaseModel):
    prompt_tokens: int = 0
    prompt_cached_tokens: int | None = None
    prompt_image_tokens: int | None = None
    completion_tokens: int = 0
    total_tokens: int

    @classmethod
    def from_raw(cls, usage: object) -> "_GoogleUsage":
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", None)
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens

        return cls(
            prompt_tokens=prompt_tokens,
            prompt_cached_tokens=getattr(usage, "cached_content_token_count", None),
            prompt_image_tokens=_google_image_token_count(usage),
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


def _google_image_token_count(usage: object) -> int | None:
    details = getattr(usage, "prompt_tokens_details", None)
    if not details:
        return None

    total = 0
    for detail in details:
        modality = str(getattr(detail, "modality", "")).lower()
        if "image" in modality:
            total += getattr(detail, "token_count", 0) or 0
    return total or None
