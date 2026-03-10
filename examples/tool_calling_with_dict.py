import asyncio
import json
from llmify import ChatOpenAI
from llmify.messages import UserMessage
from dotenv import load_dotenv

load_dotenv(override=True)


async def main():
    llm = ChatOpenAI(model="gpt-4o-mini")

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

    messages = [UserMessage(content="Wie ist das Wetter in Münster?")]
    response = await llm.invoke(messages, tools=tools)

    print(f"Content: {response.completion}")
    for tc in response.tool_calls:
        print(f"Tool: {tc.function.name}, Args: {json.loads(tc.function.arguments)}")


if __name__ == "__main__":
    asyncio.run(main())
