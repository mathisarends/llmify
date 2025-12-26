from abc import ABC, abstractmethod
from typing import AsyncIterator, TypeVar, Type
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class BaseLLM(ABC):
    @abstractmethod
    async def invoke(self, messages: list[Message]) -> str:
        pass
    
    @abstractmethod
    async def invoke_structured(
        self, 
        messages: list[Message], 
        response_model: Type[T]
    ) -> T:
        pass
    
    @abstractmethod
    async def stream(
        self, 
        messages: list[Message]
    ) -> AsyncIterator[str]:
        pass