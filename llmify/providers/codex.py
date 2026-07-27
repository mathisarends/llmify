from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Self

import httpx

from llmify.auth.codex_cli import CodexCliAuth
from llmify.providers._openai_utils import resolve_api_key
from llmify.providers.openai_responses import ChatOpenAIResponses
from llmify.utils import timed

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
        store: bool = False,
        timeout: float | httpx.Timeout | None = 60.0,
        max_retries: int = 2,
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
            store=store,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=headers,
            **kwargs,
        )

    @classmethod
    @timed("ChatCodex.from_codex_cli")
    def from_codex_cli(
        cls,
        model: str,
        *,
        auth_path: Path | None = None,
        **kwargs: Any,
    ) -> Self:
        """Build a client from the login of the locally installed Codex CLI.

        Takes account id and access token from `~/.codex/auth.json` (or
        `$CODEX_HOME/auth.json`) and keeps the token fresh for the lifetime of
        this instance. Raises `CodexCredentialsError` if the CLI is not logged
        in with a ChatGPT account.
        """
        auth = CodexCliAuth(auth_path=auth_path)
        return cls(
            model=model,
            api_key=auth,
            chatgpt_account_id=auth.account_id,
            **kwargs,
        )
