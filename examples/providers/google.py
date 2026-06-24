import asyncio
import json
import os

from dotenv import load_dotenv

from llmify import ChatGoogle, RetryableError, SystemMessage, UserMessage

load_dotenv(override=True)


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

    try:
        response = await llm.invoke(
            [
                SystemMessage(content="You are a helpful assistant."),
                UserMessage(content="What is the weather in Berlin?"),
            ],
            tools=tools,
        )
    except RetryableError as exc:
        print(f"Google request failed temporarily: {exc}")
        print("Try again later or set GOOGLE_MODEL to another Gemini model.")
        return

    print(f"Content: {response.completion}")

    for tool_call in response.tool_calls:
        print(
            f"Tool: {tool_call.function.name}, "
            f"Args: {json.loads(tool_call.function.arguments)}"
        )

    if response.usage:
        print(f"Tokens: {response.usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
