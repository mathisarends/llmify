from openai import AsyncOpenAI
from llmify.messages import Message
from llmify.base import BaseLLM
from typing import Type, TypeVar, AsyncIterator
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def invoke(self, messages: list[Message]) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self._convert_messages(messages)
        )
        return response.choices[0].message.content

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    async def invoke_structured(
        self, 
        messages: list[Message], 
        response_model: Type[T]
    ) -> T:
        response = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=self._convert_messages(messages),
            response_format=response_model
        )
        return response.choices[0].message.parsed
    
    async def stream(self, messages: list[Message]) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=self._convert_messages(messages),
            stream=True
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content