from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Self

import httpx

from llmify.auth.codex_cli import CodexCliAuth, read_codex_credentials
from llmify.providers._openai_utils import resolve_api_key
from llmify.providers.openai_responses import ChatOpenAIResponses, ReasoningEffort
from llmify.providers.openai_responses_transport import ResponsesTransport
from llmify.providers.openai_responses_types import (
    ContinuationMode,
    PromptCacheOptions,
    ReasoningSummary,
    ResponsesOptions,
)
from llmify.retries import RetryCallback

_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"


class ChatCodex(ChatOpenAIResponses):
    def __init__(
        self,
        model: str,
        chatgpt_account_id: str,
        api_key: str | Callable[[], Awaitable[str]] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        store: bool = False,
        transport: ResponsesTransport | None = None,
        responses_options: ResponsesOptions | None = None,
        continuation_mode: ContinuationMode = ContinuationMode.STATELESS,
        preserve_reasoning: bool = True,
        reasoning_summary: ReasoningSummary | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_options: PromptCacheOptions | None = None,
        timeout: float | httpx.Timeout | None = 60.0,
        max_retries: int = 2,
        on_retry: RetryCallback | None = None,
        default_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        if not chatgpt_account_id:
            raise ValueError("'chatgpt_account_id' must not be empty.")

        api_key = resolve_api_key(api_key, "CODEX_ACCESS_KEY", "Codex")
        headers = {
            **(default_headers or {}),
            "ChatGPT-Account-Id": chatgpt_account_id,
        }

        super().__init__(
            model=model,
            api_key=api_key,
            base_url=_CODEX_BASE_URL,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            store=store,
            transport=transport,
            responses_options=responses_options,
            continuation_mode=continuation_mode,
            preserve_reasoning=preserve_reasoning,
            reasoning_summary=reasoning_summary,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_options=prompt_cache_options,
            timeout=timeout,
            max_retries=max_retries,
            on_retry=on_retry,
            default_headers=headers,
            **kwargs,
        )

    @classmethod
    def from_cli(
        cls,
        model: str,
        *,
        auth_path: Path | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        store: bool = False,
        transport: ResponsesTransport | None = None,
        responses_options: ResponsesOptions | None = None,
        continuation_mode: ContinuationMode = ContinuationMode.STATELESS,
        preserve_reasoning: bool = True,
        reasoning_summary: ReasoningSummary | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_options: PromptCacheOptions | None = None,
        timeout: float | httpx.Timeout | None = 60.0,
        max_retries: int = 2,
        on_retry: RetryCallback | None = None,
        default_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Build a client from the login of the locally installed Codex CLI.

        Takes the same model options as the constructor; `api_key` and
        `chatgpt_account_id` come from the CLI login instead. `auth_path` points
        at a different `auth.json` than the default.

        Reads account id and access token from `~/.codex/auth.json` (or
        `$CODEX_HOME/auth.json`); nothing is sent or written here. The token is
        then kept fresh for the lifetime of this instance, refreshed from the
        request path as it approaches expiry. Raises `CodexCredentialsError` if
        the CLI is not logged in with a ChatGPT account.
        """
        codex_credentials = read_codex_credentials(auth_path=auth_path)
        auth = CodexCliAuth(codex_credentials)
        return cls(
            model=model,
            api_key=auth,
            chatgpt_account_id=auth.account_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            store=store,
            transport=transport,
            responses_options=responses_options,
            continuation_mode=continuation_mode,
            preserve_reasoning=preserve_reasoning,
            reasoning_summary=reasoning_summary,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_options=prompt_cache_options,
            timeout=timeout,
            max_retries=max_retries,
            on_retry=on_retry,
            default_headers=default_headers,
            **kwargs,
        )
