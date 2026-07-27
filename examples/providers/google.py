import asyncio
import json
import os

from dotenv import load_dotenv

from llmify import (
    AssistantMessage,
    ChatGoogle,
    RetryableError,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)

load_dotenv(override=True)


def get_weather(city: str, unit: str = "celsius") -> str:
    temperature = 18 if unit == "celsius" else 64
    return json.dumps(
        {
            "city": city,
            "temperature": temperature,
            "unit": unit,
            "conditions": "partly cloudy",
        }
    )


async def main():
    llm = ChatGoogle(model=os.getenv("GOOGLE_MODEL", "gemini-3.5-flash"))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "default": "celsius",
                        },
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What is the weather in Berlin?"),
    ]

    try:
        response = await llm.invoke(messages, tools=tools)

        if not response.tool_calls:
            print(f"Content: {response.completion}")
            return

        messages.append(
            AssistantMessage(
                content=response.completion or None,
                tool_calls=response.tool_calls,
            )
        )
        for tool_call in response.tool_calls:
            args = json.loads(tool_call.function.arguments)
            print(f"Tool: {tool_call.function.name}, Args: {args}")

            if tool_call.function.name != "get_weather":
                raise ValueError(f"Unknown tool: {tool_call.function.name}")

            result = get_weather(**args)
            messages.append(
                ToolResultMessage(
                    tool_call_id=tool_call.id,
                    content=result,
                )
            )

        final_response = await llm.invoke(messages, tools=tools)
        print(f"Content: {final_response.completion}")

        total_tokens = sum(
            item.usage.total_tokens
            for item in (response, final_response)
            if item.usage is not None
        )
        print(f"Tokens: {total_tokens}")
    except RetryableError as exc:
        print(f"Google request failed temporarily: {exc}")
        print("Try again later or set GOOGLE_MODEL to another Gemini model.")


if __name__ == "__main__":
    asyncio.run(main())
