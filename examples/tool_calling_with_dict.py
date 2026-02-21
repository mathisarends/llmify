import asyncio
from llmify import ChatOpenAI
from llmify.messages import UserMessage


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

    messages = [UserMessage("Wie ist das Wetter in Münster?")]
    response = await llm.invoke(messages, tools=tools)

    print(f"Content: {response.content}")
    for tc in response.tool_calls:
        print(f"Tool: {tc.name}, Args: {tc.tool}")


if __name__ == "__main__":
    asyncio.run(main())
