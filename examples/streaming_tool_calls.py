import asyncio
import json

from dotenv import load_dotenv

from llmify import ChatOpenAI, SystemMessage, UserMessage, tool


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get weather for a city."""
    return f"The weather in {city} is 22 degrees {unit}."


async def main() -> None:
    load_dotenv(override=True)

    llm = ChatOpenAI(model="gpt-4o")
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content="What is the weather in Berlin? Then answer with a spoken text with a greeting and the weather, and use the get_weather tool to get the weather information."
        ),
    ]

    async for event in llm.stream(messages, tools=[get_weather]):
        match event.type:
            case "text":
                print(event.delta, end="", flush=True)
            case "tool_call":
                tool_call = event.tool_call
                if tool_call.function.name == "get_weather":
                    args = json.loads(tool_call.function.arguments)
                    result = get_weather(**args)
                    print(f"\n[tool:{tool_call.function.name}] {result}")
            case "end":
                usage = event.usage.total_tokens if event.usage else "unknown"
                print(f"\n[stop={event.stop_reason}, tokens={usage}]")


if __name__ == "__main__":
    asyncio.run(main())
