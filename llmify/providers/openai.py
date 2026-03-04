import os
import httpx
from openai import AsyncOpenAI
from llmify.providers._base_openai import BaseOpenAICompatible
from typing import Any


class ChatOpenAI(BaseOpenAICompatible):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
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
        default_headers: dict | None = None,
        **kwargs: Any,
    ):
        super().__init__(
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
        self._client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )
        self._model = model
