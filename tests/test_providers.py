from llmify.providers.base import BaseOpenAICompatible
from llmify import UserMessage, ImageMessage


class DummyOpenAI(BaseOpenAICompatible):
    pass


def test_convert_simple_messages():
    provider = DummyOpenAI()
    messages = [UserMessage("Hello")]

    converted = provider._convert_messages(messages)

    assert converted == [{"role": "user", "content": "Hello"}]


def test_convert_image_message():
    provider = DummyOpenAI()
    messages = [
        ImageMessage(base64_data="abc123", media_type="image/png", text="What is this?")
    ]

    converted = provider._convert_messages(messages)

    assert converted[0]["role"] == "user"
    assert len(converted[0]["content"]) == 2
    assert converted[0]["content"][0] == {"type": "text", "text": "What is this?"}
    assert converted[0]["content"][1]["type"] == "image_url"
    assert (
        "data:image/png;base64,abc123" in converted[0]["content"][1]["image_url"]["url"]
    )
