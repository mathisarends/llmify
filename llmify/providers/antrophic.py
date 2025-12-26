import os
import httpx
from anthropic import AsyncAnthropic
from llmify.messages import Message, ImageMessage, SystemMessage
from llmify.providers.base import BaseChatModel
from typing import Type, TypeVar, AsyncIterator, Any
from pydantic import BaseModel
from dotenv import load_dotenv

T = TypeVar('T', bound=BaseModel)

load_dotenv(override=True)

class ChatAnthropic(BaseChatModel):
    def __init__(
        self, 
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int | None = 4096,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        timeout: float | httpx.Timeout | None = 60.0,
        max_retries: int = 2,
        **kwargs: Any
    ):
        super().__init__(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        self._client = AsyncAnthropic(
            api_key=api_key,
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
        system_msg, converted_messages = self._convert_messages(messages)
        params = self._merge_params({"max_tokens": max_tokens, "temperature": temperature, **kwargs})
        
        response = await self._client.messages.create(
            model=self._model,
            system=system_msg,
            messages=converted_messages,
            **params
        )
        return response.content[0].text

    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[dict]]:
        system_content = ""
        converted = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_content = msg.content
                continue
            
            if isinstance(msg, ImageMessage):
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": msg.media_type,
                        "data": msg.base64_data
                    }
                })
                converted.append({"role": msg.role, "content": content})
                continue
            
            converted.append({"role": msg.role, "content": msg.content})
        
        return system_content, converted
    
    async def invoke_structured(
        self, 
        messages: list[Message], 
        response_model: Type[T],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any
    ) -> T:
        system_msg, converted_messages = self._convert_messages(messages)
        params = self._merge_params({"max_tokens": max_tokens, "temperature": temperature, **kwargs})
        
        tools = [{
            "name": "output",
            "description": "Output structured data",
            "input_schema": response_model.model_json_schema()
        }]
        
        response = await self._client.messages.create(
            model=self._model,
            system=system_msg,
            messages=converted_messages,
            tools=tools,
            tool_choice={"type": "tool", "name": "output"},
            **params
        )
        
        tool_use = next(block for block in response.content if block.type == "tool_use")
        return response_model.model_validate(tool_use.input)
    
    async def stream(
        self, 
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        system_msg, converted_messages = self._convert_messages(messages)
        params = self._merge_params({"max_tokens": max_tokens, "temperature": temperature, **kwargs})
        
        async with self._client.messages.stream(
            model=self._model,
            system=system_msg,
            messages=converted_messages,
            **params
        ) as stream:
            async for text in stream.text_stream:
                yield text