import pytest
from pydantic import BaseModel

pytest.importorskip("google.genai")

from llmify.messages import (
    AssistantMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    SystemMessage,
    UserMessage,
)
from llmify.providers.google import (
    _build_config,
    _convert_messages,
    _convert_tool,
    _convert_user_parts,
)

WEATHER_TOOL = {
    "function": {
        "name": "get_weather",
        "description": "Look up the weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
}


class TestBuildConfig:
    def test_translates_openai_style_params(self) -> None:
        config = _build_config(
            {
                "max_tokens": 512,
                "temperature": 0.2,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.3,
                "seed": 7,
            }
        )

        assert config is not None
        assert config.max_output_tokens == 512
        assert config.temperature == 0.2
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.1
        assert config.presence_penalty == 0.3
        assert config.seed == 7

    def test_wraps_a_single_stop_string_in_a_list(self) -> None:
        config = _build_config({"stop": "END"})
        assert config is not None
        assert config.stop_sequences == ["END"]

    def test_keeps_a_stop_list_as_is(self) -> None:
        config = _build_config({"stop": ["END", "STOP"]})
        assert config is not None
        assert config.stop_sequences == ["END", "STOP"]

    def test_returns_none_when_nothing_is_configured(self) -> None:
        assert _build_config({}) is None

    def test_forwards_unknown_params_untouched(self) -> None:
        config = _build_config({"top_k": 40})
        assert config is not None
        assert config.top_k == 40

    def test_merges_a_response_format_dict(self) -> None:
        config = _build_config(
            {"response_format": {"response_mime_type": "application/json"}}
        )
        assert config is not None
        assert config.response_mime_type == "application/json"

    def test_sets_the_system_instruction(self) -> None:
        config = _build_config({}, system_instruction="You are terse.")
        assert config is not None
        assert config.system_instruction == "You are terse."


class TestBuildConfigStructuredOutput:
    class Answer(BaseModel):
        city: str
        degrees: int

    def test_requests_json_matching_the_model_schema(self) -> None:
        config = _build_config({}, output_format=self.Answer)

        assert config is not None
        assert config.response_mime_type == "application/json"
        assert config.response_json_schema == self.Answer.model_json_schema()


class TestBuildConfigTools:
    def test_registers_function_declarations(self) -> None:
        config = _build_config({}, tools=[WEATHER_TOOL])

        assert config is not None
        assert config.tools is not None
        declarations = config.tools[0].function_declarations
        assert declarations is not None
        assert [decl.name for decl in declarations] == ["get_weather"]

    def test_disables_the_sdk_function_calling_loop(self) -> None:
        # llmify hands tool calls back to the caller, so the SDK must not
        # execute them itself.
        config = _build_config({}, tools=[WEATHER_TOOL])

        assert config is not None
        assert config.automatic_function_calling is not None
        assert config.automatic_function_calling.disable is True

    @pytest.mark.parametrize(
        ("tool_choice", "expected_mode"), [("auto", "AUTO"), ("required", "ANY")]
    )
    def test_maps_tool_choice_to_function_calling_mode(
        self, tool_choice: str, expected_mode: str
    ) -> None:
        config = _build_config({}, tools=[WEATHER_TOOL], tool_choice=tool_choice)

        assert config is not None
        assert config.tool_config is not None
        assert config.tool_config.function_calling_config is not None
        assert config.tool_config.function_calling_config.mode == expected_mode

    def test_drops_the_tools_when_tool_choice_is_none(self) -> None:
        config = _build_config(
            {"temperature": 0.1}, tools=[WEATHER_TOOL], tool_choice="none"
        )

        assert config is not None
        assert not config.tools


class TestConvertTool:
    def test_translates_an_openai_function_schema(self) -> None:
        assert _convert_tool(WEATHER_TOOL) == {
            "name": "get_weather",
            "description": "Look up the weather",
            "parameters_json_schema": WEATHER_TOOL["function"]["parameters"],
        }

    def test_accepts_a_bare_function_object(self) -> None:
        converted = _convert_tool({"name": "ping", "parameters": {"type": "object"}})

        assert converted["name"] == "ping"
        assert converted["description"] == ""
        assert converted["parameters_json_schema"] == {"type": "object"}

    def test_passes_through_native_google_declarations(self) -> None:
        native = {"name": "ping", "parameters_json_schema": {"type": "object"}}
        assert _convert_tool(native) is native

    def test_rejects_a_non_object_function_field(self) -> None:
        with pytest.raises(TypeError, match="function object"):
            _convert_tool({"function": "get_weather"})


class TestConvertUserParts:
    def test_wraps_plain_string_content(self) -> None:
        parts = _convert_user_parts(UserMessage(content="Hello"))
        assert parts == [{"text": "Hello"}]

    def test_inlines_base64_data_urls(self) -> None:
        message = UserMessage(
            content=[
                ContentPartTextParam(text="What is this?"),
                ContentPartImageParam(
                    image_url=ImageURL(url="data:image/png;base64,abc123")
                ),
            ]
        )

        assert _convert_user_parts(message) == [
            {"text": "What is this?"},
            {"inline_data": {"mime_type": "image/png", "data": "abc123"}},
        ]

    def test_references_remote_images_by_uri(self) -> None:
        message = UserMessage(
            content=[
                ContentPartImageParam(
                    image_url=ImageURL(
                        url="https://example.com/cat.jpeg", media_type="image/jpeg"
                    )
                )
            ]
        )

        assert _convert_user_parts(message) == [
            {
                "file_data": {
                    "mime_type": "image/jpeg",
                    "file_uri": "https://example.com/cat.jpeg",
                }
            }
        ]


class TestConvertMessages:
    def test_lifts_the_system_message_out_of_the_contents(self) -> None:
        contents, system_instruction = _convert_messages(
            [
                SystemMessage(content="You are terse."),
                UserMessage(content="Hi"),
            ]
        )

        assert system_instruction == "You are terse."
        assert contents == [{"role": "user", "parts": [{"text": "Hi"}]}]

    def test_maps_the_assistant_role_to_model(self) -> None:
        contents, _ = _convert_messages(
            [UserMessage(content="Hi"), AssistantMessage(content="Hello")]
        )

        assert [content["role"] for content in contents] == ["user", "model"]
        assert contents[1]["parts"] == [{"text": "Hello"}]

    def test_returns_no_system_instruction_when_absent(self) -> None:
        _, system_instruction = _convert_messages([UserMessage(content="Hi")])
        assert system_instruction is None
