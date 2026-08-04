# tests/tools/test_function_tool.py
from typing import Annotated

import pytest
from pydantic import Field

from llmify.tools import FunctionTool, tool


def sample_function(query: str, max_results: int = 10) -> str:
    """Search for information"""
    return f"Searching: {query}"


def no_hints_function(query, count=5):
    return f"{query}: {count}"


class TestFunctionTool:
    def test_extracts_name_from_function(self) -> None:
        tool = FunctionTool(sample_function)
        assert tool.name == "sample_function"

    def test_uses_custom_name_when_provided(self) -> None:
        tool = FunctionTool(sample_function, name="custom_search")
        assert tool.name == "custom_search"

    def test_extracts_description_from_docstring(self) -> None:
        tool = FunctionTool(sample_function)
        schema = tool.to_openai_schema()
        assert schema["function"]["description"] == "Search for information"

    def test_generates_valid_openai_schema(self) -> None:
        tool = FunctionTool(sample_function)
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "sample_function"
        assert "parameters" in schema["function"]

    def test_identifies_required_parameters(self) -> None:
        tool = FunctionTool(sample_function)
        params = tool.to_openai_schema()["function"]["parameters"]

        assert params["required"] == ["query"]
        assert "max_results" not in params["required"]

    def test_maps_python_types_to_json_schema(self) -> None:
        tool = FunctionTool(sample_function)
        props = tool.to_openai_schema()["function"]["parameters"]["properties"]

        assert props["query"]["type"] == "string"
        assert props["max_results"]["type"] == "integer"

    def test_defaults_to_string_when_no_type_hints(self) -> None:
        tool = FunctionTool(no_hints_function)
        props = tool.to_openai_schema()["function"]["parameters"]["properties"]

        assert props["query"]["type"] == "string"
        assert props["count"]["type"] == "string"

    def test_parses_json_arguments(self) -> None:
        tool = FunctionTool(sample_function)
        args = tool.parse_arguments('{"query": "test", "max_results": 5}')

        assert args == {"query": "test", "max_results": 5}

    def test_remains_callable(self) -> None:
        tool = FunctionTool(sample_function)
        result = tool("test query", max_results=3)

        assert result == "Searching: test query"

    def test_ignores_self_and_cls_parameters(self) -> None:
        class MyClass:
            def method(self, query: str) -> str:
                return query

        tool = FunctionTool(MyClass().method)
        props = tool.to_openai_schema()["function"]["parameters"]["properties"]

        assert "self" not in props
        assert "query" in props


class TestTypeMapping:
    @pytest.mark.parametrize(
        ("annotation", "expected"),
        [
            (str, "string"),
            (int, "integer"),
            (float, "number"),
            (bool, "boolean"),
            (list, "array"),
            (dict, "object"),
        ],
    )
    def test_maps_builtin_types(self, annotation: type, expected: str) -> None:
        def fn(value):
            return value

        fn.__annotations__ = {"value": annotation}

        props = FunctionTool(fn).to_openai_schema()["function"]["parameters"][
            "properties"
        ]
        assert props["value"]["type"] == expected

    def test_maps_parameterised_list_to_array(self) -> None:
        def fn(tags: list[str]) -> str:
            return ", ".join(tags)

        props = FunctionTool(fn).to_openai_schema()["function"]["parameters"][
            "properties"
        ]
        assert props["tags"] == {"type": "array"}

    def test_maps_parameterised_dict_to_object(self) -> None:
        def fn(options: dict[str, int]) -> int:
            return sum(options.values())

        props = FunctionTool(fn).to_openai_schema()["function"]["parameters"][
            "properties"
        ]
        assert props["options"] == {"type": "object"}

    def test_falls_back_to_string_for_unknown_types(self) -> None:
        class Custom:
            pass

        def fn(value: Custom) -> str:
            return str(value)

        props = FunctionTool(fn).to_openai_schema()["function"]["parameters"][
            "properties"
        ]
        assert props["value"] == {"type": "string"}

    def test_survives_unresolvable_annotations(self) -> None:
        # A forward reference to a name that never materialises makes
        # get_type_hints raise; the schema must still be generated.
        def fn(value: "NeverDefined") -> str:  # noqa: F821
            return value

        props = FunctionTool(fn).to_openai_schema()["function"]["parameters"][
            "properties"
        ]
        assert props["value"] == {"type": "string"}


class TestAnnotatedDescriptions:
    def test_picks_up_a_field_description(self) -> None:
        def fn(city: Annotated[str, Field(description="City to look up")]) -> str:
            return city

        props = FunctionTool(fn).to_openai_schema()["function"]["parameters"][
            "properties"
        ]
        assert props["city"]["description"] == "City to look up"
        assert props["city"]["type"] == "string"

    def test_ignores_metadata_without_a_description(self) -> None:
        def fn(count: Annotated[int, "just a note"]) -> int:
            return count

        props = FunctionTool(fn).to_openai_schema()["function"]["parameters"][
            "properties"
        ]
        assert props["count"] == {"type": "integer"}

    def test_does_not_leak_a_description_between_parameters(self) -> None:
        def fn(
            city: Annotated[str, Field(description="City to look up")],
            country: str,
        ) -> str:
            return f"{city}, {country}"

        props = FunctionTool(fn).to_openai_schema()["function"]["parameters"][
            "properties"
        ]
        assert "description" not in props["country"]


class TestToolDecorator:
    def test_converts_function_to_tool(self) -> None:
        @tool
        def search(query: str) -> str:
            """Search function"""
            return query

        assert isinstance(search, FunctionTool)
        assert search.name == "search"

    def test_accepts_custom_name(self) -> None:
        @tool(name="web_search")
        def search(query: str) -> str:
            return query

        assert search.name == "web_search"

    def test_accepts_custom_description(self) -> None:
        @tool(description="Custom desc")
        def search(query: str) -> str:
            """Original doc"""
            return query

        schema = search.to_openai_schema()
        assert schema["function"]["description"] == "Custom desc"

    def test_preserves_function_behavior(self) -> None:
        @tool
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5
