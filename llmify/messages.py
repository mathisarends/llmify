from dataclasses import dataclass
from enum import StrEnum

class _MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: _MessageRole
    content: str

class SystemMessage(Message):
    def __init__(self, content: str):
        super().__init__(role=_MessageRole.SYSTEM, content=content)

class UserMessage(Message):
    def __init__(self, content: str):
        super().__init__(role=_MessageRole.USER, content=content)

class AssistantMessage(Message):
    def __init__(self, content: str):
        super().__init__(role=_MessageRole.ASSISTANT, content=content)