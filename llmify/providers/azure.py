import os
from typing import Any, cast

import httpx

try:
    from openai import AsyncAzureOpenAI
except ImportError:
    raise ImportError(
        "The 'openai' package is required for ChatAzureOpenAI. "
        "Install it with: pip install py-llmify[openai]"
    )

from llmify.providers.openai_compatible import OpenAICompatible
from llmify.retries import RetryCallback


class ChatAzureOpenAI(OpenAICompatible):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str = "2024-02-15-preview",
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
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if azure_endpoint is None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=cast(str, azure_endpoint),
            api_version=api_version,
            timeout=timeout,
            max_retries=0,
        )
