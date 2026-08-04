import pytest

from llmify import (
    AssistantMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    Function,
    ImageURL,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from llmify.messages import ContentPartRefusalParam, _truncate

BASE64_PNG = "data:image/png;base64," + "A" * 4096


class TestUserMessage:
    def test_creates_with_string_content(self) -> None:
        msg = UserMessage(content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_text_property_returns_string_content(self) -> None:
        msg = UserMessage(content="Hello")
        assert msg.text == "Hello"


class TestSystemMessage:
    def test_creates_with_string_content(self) -> None:
        msg = SystemMessage(content="You are helpful")
        assert msg.role == "system"
        assert msg.content == "You are helpful"


class TestUserMessageWithImageContent:
    def test_creates_with_text_and_image_parts(self) -> None:
        img = ContentPartImageParam(
            image_url=ImageURL(
                url="data:image/png;base64,abc123", media_type="image/png"
            )
        )
        msg = UserMessage(content=[ContentPartTextParam(text="What is this?"), img])
        assert msg.role == "user"
        assert len(msg.content) == 2

    def test_text_property_extracts_text_parts(self) -> None:
        img = ContentPartImageParam(
            image_url=ImageURL(
                url="data:image/png;base64,abc123", media_type="image/png"
            )
        )
        msg = UserMessage(content=[ContentPartTextParam(text="What is this?"), img])
        assert msg.text == "What is this?"


class TestImageURL:
    def test_stores_media_type(self) -> None:
        img_url = ImageURL(url="data:image/png;base64,abc123", media_type="image/png")
        assert img_url.media_type == "image/png"

    def test_default_media_type_is_png(self) -> None:
        img_url = ImageURL(url="data:image/png;base64,abc123")
        assert img_url.media_type == "image/png"

    def test_default_detail_is_auto(self) -> None:
        png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        img_url = ImageURL(
            url=f"data:image/png;base64,{png_base64}", media_type="image/png"
        )
        assert img_url.detail == "auto"


class TestTruncate:
    def test_leaves_short_text_untouched(self) -> None:
        assert _truncate("short") == "short"

    def test_leaves_text_at_the_limit_untouched(self) -> None:
        assert _truncate("x" * 50) == "x" * 50

    def test_caps_longer_text_at_the_limit(self) -> None:
        truncated = _truncate("x" * 100)
        assert len(truncated) == 50
        assert truncated.endswith("...")

    def test_honours_a_custom_limit(self) -> None:
        assert _truncate("abcdefghij", 5) == "ab..."


class TestAssistantMessageText:
    def test_returns_string_content(self) -> None:
        assert AssistantMessage(content="Hello").text == "Hello"

    def test_concatenates_text_parts(self) -> None:
        message = AssistantMessage(
            content=[ContentPartTextParam(text="a"), ContentPartTextParam(text="b")]
        )
        assert message.text == "ab"

    def test_marks_up_refusal_parts(self) -> None:
        message = AssistantMessage(
            content=[
                ContentPartTextParam(text="Sure. "),
                ContentPartRefusalParam(refusal="I cannot help with that."),
            ]
        )
        assert message.text == "Sure. [Refusal] I cannot help with that."

    def test_returns_empty_string_for_none_content(self) -> None:
        assert AssistantMessage().text == ""


class TestSystemMessageText:
    def test_concatenates_text_parts_with_newlines(self) -> None:
        message = SystemMessage(
            content=[ContentPartTextParam(text="a"), ContentPartTextParam(text="b")]
        )
        assert message.text == "a\nb"


class TestReprDoesNotLeakBase64:
    """Message reprs land in logs and tracebacks, so images must stay compact."""

    def test_image_url_str_collapses_a_data_url(self) -> None:
        rendered = str(ImageURL(url=BASE64_PNG))
        assert "<base64 image/png>" in rendered
        assert "AAAA" not in rendered

    def test_image_url_repr_collapses_a_data_url(self) -> None:
        assert "AAAA" not in repr(ImageURL(url=BASE64_PNG))

    def test_content_part_delegates_to_the_image_url(self) -> None:
        part = ContentPartImageParam(image_url=ImageURL(url=BASE64_PNG))
        assert "AAAA" not in str(part)
        assert "AAAA" not in repr(part)

    def test_keeps_short_remote_urls_readable(self) -> None:
        rendered = str(ImageURL(url="https://example.com/cat.png"))
        assert "https://example.com/cat.png" in rendered

    def test_truncates_long_remote_urls(self) -> None:
        rendered = str(ImageURL(url="https://example.com/" + "a" * 200 + ".png"))
        assert rendered.endswith("...")


class TestReprsAreReadable:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            (UserMessage(content="Hi"), "UserMessage"),
            (SystemMessage(content="Be terse"), "SystemMessage"),
            (AssistantMessage(content="Hello"), "AssistantMessage"),
            (
                ToolResultMessage(tool_call_id="call_1", content="18"),
                "ToolResultMessage",
            ),
        ],
    )
    def test_names_the_message_type(self, message: object, expected: str) -> None:
        assert str(message).startswith(expected)
        assert repr(message).startswith(expected)

    def test_tool_call_shows_the_function_and_arguments(self) -> None:
        call = ToolCall(
            id="call_1",
            function=Function(name="get_weather", arguments='{"city": "Berlin"}'),
        )

        assert "call_1" in str(call)
        assert "get_weather" in str(call)
        assert "get_weather" in repr(call)

    def test_function_str_previews_the_arguments(self) -> None:
        function = Function(name="get_weather", arguments='{"city": "Berlin"}')
        assert str(function) == 'get_weather({"city": "Berlin"})'

    def test_text_part_str_labels_the_content(self) -> None:
        part = ContentPartTextParam(text="Hello")
        assert str(part) == "Text: Hello"
        assert "Hello" in repr(part)

    def test_refusal_part_str_labels_the_content(self) -> None:
        part = ContentPartRefusalParam(refusal="I cannot help")
        assert str(part) == "Refusal: I cannot help"
        assert "I cannot help" in repr(part)

    def test_tool_result_truncates_long_content(self) -> None:
        message = ToolResultMessage(tool_call_id="call_1", content="x" * 200)
        assert len(str(message)) < 150
