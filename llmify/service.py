import json
from typing import Any
from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema

from llmify.tools.models import ToolCall


class ToolSchemaGenerator(GenerateJsonSchema):
    def generate_inner(self, schema: core_schema.CoreSchema) -> JsonSchemaValue:
        json_schema = super().generate_inner(schema)
        if isinstance(json_schema, dict):
            json_schema.pop("title", None)
        return json_schema


def pydantic_to_tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to OpenAI tool schema format"""

    schema = model.model_json_schema(
        schema_generator=ToolSchemaGenerator, mode="validation"
    )

    # Remove unnecessary fields
    schema.pop("title", None)
    schema.pop("additionalProperties", None)

    return {
        "type": "function",
        "function": {
            "name": model.__name__,
            "description": model.__doc__ or "",
            "parameters": schema,
        },
    }


def parse_tool_call(
    tool_call_id: str,
    function_name: str,
    arguments: str,
    tools: dict[str, type[BaseModel]],
) -> ToolCall:
    """Parse OpenAI tool call into a ToolCall with Pydantic instance"""

    if function_name not in tools:
        raise ValueError(f"Unknown tool: {function_name}")

    tool_class = tools[function_name]
    args_dict = json.loads(arguments)
    tool_instance = tool_class(**args_dict)

    return ToolCall(id=tool_call_id, name=function_name, tool=tool_instance)
