from llmify import UserMessage, SystemMessage, ImageMessage


def test_user_message():
    msg = UserMessage("Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_system_message():
    msg = SystemMessage("You are helpful")
    assert msg.role == "system"
    assert msg.content == "You are helpful"


def test_image_message_with_text():
    msg = ImageMessage(base64_data="abc123", text="What is this?")
    assert msg.role == "user"
    assert msg.content == "What is this?"
    assert msg.base64_data == "abc123"


def test_image_message_auto_detect_png():
    png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    msg = ImageMessage(base64_data=png_base64)
    assert msg.media_type == "image/png"
