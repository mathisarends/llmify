import os
import httpx
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletionChunk
from llmify.messages import Message
from llmify.providers.base import BaseOpenAICompatible
from typing import Any, Type, TypeVar, AsyncIterator
from pydantic import BaseModel
from dotenv import load_dotenv

T = TypeVar('T', bound=BaseModel)

load_dotenv(override=True)

class ChatAzureOpenAI(BaseOpenAICompatible):
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
        **kwargs: Any
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
            **kwargs
        )
        if api_key is None:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if azure_endpoint is None:
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=timeout,
            max_retries=max_retries
        )
        self._model = model
    
    async def invoke(
        self, 
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any
    ) -> str:
        params = self._merge_params({"max_tokens": max_tokens, "temperature": temperature, **kwargs})
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=self._convert_messages(messages),
            **params
        )
        return response.choices[0].message.content
    
    async def invoke_structured(
        self, 
        messages: list[Message], 
        response_model: Type[T],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any
    ) -> T:
        params = self._merge_params({"max_tokens": max_tokens, "temperature": temperature, **kwargs})
        response = await self._client.beta.chat.completions.parse(
            model=self._model,
            messages=self._convert_messages(messages),
            response_format=response_model,
            **params
        )
        return response.choices[0].message.parsed
    
    async def stream(
        self, 
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        params = self._merge_params({"max_tokens": max_tokens, "temperature": temperature, **kwargs})
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=self._convert_messages(messages),
            stream=True,
            **params
        )
        chunk: ChatCompletionChunk
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content