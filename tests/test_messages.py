from llmify import UserMessage, SystemMessage, ContentPartTextParam, ContentPartImageParam, ImageURL


def test_user_message():
    msg = UserMessage(content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_system_message():
    msg = SystemMessage(content="You are helpful")
    assert msg.role == "system"
    assert msg.content == "You are helpful"


def test_user_message_with_image_and_text():
    msg = UserMessage(content=[
        ContentPartTextParam(text="What is this?"),
        ContentPartImageParam(
            image_url=ImageURL(url="data:image/png;base64,abc123", media_type="image/png")
        ),
    ])
    assert msg.role == "user"
    assert msg.text == "What is this?"
    assert isinstance(msg.content, list)
    assert len(msg.content) == 2
    assert msg.content[1].image_url.media_type == "image/png"


def test_image_url_base64_display():
    url = ImageURL(url="data:image/png;base64,abc123", media_type="image/png")
    assert "base64" in str(url)
