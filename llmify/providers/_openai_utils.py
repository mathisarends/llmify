import os
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

try:
    from openai import (
        APIConnectionError as _OpenAIConnectionError,
    )
    from openai import APIError as _OpenAIAPIError
    from openai import (
        APIStatusError as _OpenAIStatusError,
    )
    from openai import (
        APITimeoutError as _OpenAITimeoutError,
    )
    from openai import (
        RateLimitError as _OpenAIRateLimitError,
    )
    from openai.types import CompletionUsage
    from openai.types.responses import ResponseUsage
except ImportError:
    if TYPE_CHECKING:
        raise

from llmify.exceptions import (
    ContextLengthExceededError,
    CredentialsUnavailableError,
    OutOfCreditsError,
    RateLimitError,
    RetryableError,
)
from llmify.messages import Function, ToolCall
from llmify.tools import Tool
from llmify.views import ChatInvokeUsage

_TRANSIENT_API_ERROR_CODES = frozenset(
    {
        "overloaded_error",
        "rate_limit_exceeded",
        "server_error",
        "timeout",
    }
)


def resolve_api_key(
    api_key: str | Callable[[], Awaitable[str]] | None,
    environment_variable: str,
    provider: str,
) -> str | Callable[[], Awaitable[str]]:
    resolved = api_key if api_key is not None else os.getenv(environment_variable)
    if resolved is None:
        raise CredentialsUnavailableError(
            f"No {provider} API key found. Pass 'api_key' or set "
            f"{environment_variable}."
        )
    return resolved


def tool_schemas(tools: list[Tool | dict]) -> list[dict]:
    return [
        tool if isinstance(tool, dict) else tool.to_openai_schema() for tool in tools
    ]


def tool_call(call_id: str, name: str, arguments: str | None) -> ToolCall:
    return ToolCall(
        id=call_id,
        function=Function(name=name, arguments=arguments or "{}"),
    )


def parse_usage(
    usage: CompletionUsage | ResponseUsage | None,
) -> ChatInvokeUsage | None:
    if usage is None:
        return None

    if isinstance(usage, CompletionUsage):
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        cached_tokens = (
            usage.prompt_tokens_details.cached_tokens
            if usage.prompt_tokens_details is not None
            else None
        )
    else:
        prompt_tokens = usage.input_tokens
        completion_tokens = usage.output_tokens
        cached_tokens = usage.input_tokens_details.cached_tokens

    return ChatInvokeUsage(
        prompt_tokens=prompt_tokens,
        prompt_cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        total_tokens=usage.total_tokens,
    )


def reject_stream_parameter(params: dict[str, Any]) -> None:
    if "stream" in params:
        raise TypeError(
            "'stream' is managed by invoke() and stream(); do not pass it as "
            "a model parameter."
        )


def map_openai_error(exc: Exception) -> Exception:
    if isinstance(exc, _OpenAIRateLimitError):
        body = exc.body or {}
        error = body.get("error") if isinstance(body, dict) else None
        code = error.get("code", "") if isinstance(error, dict) else ""
        if code == "insufficient_quota":
            return OutOfCreditsError(str(exc))
        retry_after: float | None = None
        raw = exc.response.headers.get("retry-after")
        if raw:
            try:
                retry_after = float(raw)
            except ValueError:
                pass
        return RateLimitError(str(exc), retry_after=retry_after)
    if isinstance(exc, _OpenAIStatusError) and exc.status_code == 400:
        body = exc.body or {}
        error = body.get("error") if isinstance(body, dict) else None
        code = error.get("code", "") if isinstance(error, dict) else ""
        if code == "context_length_exceeded":
            return ContextLengthExceededError(str(exc))
    if isinstance(exc, _OpenAIStatusError) and exc.status_code == 401:
        return CredentialsUnavailableError(str(exc))
    if isinstance(exc, (_OpenAIConnectionError, _OpenAITimeoutError)):
        return RetryableError(str(exc))
    if isinstance(exc, _OpenAIStatusError) and (
        exc.status_code in {408, 409} or exc.status_code >= 500
    ):
        return RetryableError(str(exc), status_code=exc.status_code)
    if type(exc) is _OpenAIAPIError:
        code = exc.code or exc.type
        message = str(exc).lower()
        if code == "rate_limit_exceeded":
            return RateLimitError(str(exc))
        if code in _TRANSIENT_API_ERROR_CODES or any(
            phrase in message
            for phrase in (
                "overloaded",
                "temporarily unavailable",
                "try again later",
            )
        ):
            return RetryableError(str(exc))
    return exc


@contextmanager
def openai_errors():
    try:
        yield
    except Exception as exc:
        mapped = map_openai_error(exc)
        if mapped is not exc:
            raise mapped from exc
        raise
