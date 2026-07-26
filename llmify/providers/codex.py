from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from llmify.providers._openai_utils import resolve_api_key
from llmify.providers.openai_responses import OpenAIResponsesAPICompatible

_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"


class ChatCodex(OpenAIResponsesAPICompatible):
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
