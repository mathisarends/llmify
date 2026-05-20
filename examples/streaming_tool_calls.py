import asyncio
import json

from dotenv import load_dotenv

from llmify import ChatOpenAI, StreamEventType, SystemMessage, UserMessage, tool


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get weather for a city."""
    return f"The weather in {city} is 22 degrees {unit}."


async def main() -> None:
    load_dotenv(override=True)

    llm = ChatOpenAI()
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content="What is the weather in Berlin? Then answer with a spoken text with a greeting and the weather, and use the get_weather tool to get the weather information."
        ),
    ]

    print("Streaming text chunks (numbered):")
    chunk_count = 0

    async for event in llm.stream(messages, tools=[get_weather]):
        match event.type:
            case StreamEventType.TEXT:
                chunk_count += 1
                print(f"[{chunk_count:02d}]{event.delta}", end="", flush=True)
            case StreamEventType.TOOL_CALL:
                tool_call = event.tool_call
                if tool_call.function.name == "get_weather":
                    args = json.loads(tool_call.function.arguments)
                    result = get_weather(**args)
                    print(f"\n[tool:{tool_call.function.name} args={args}] {result}")
            case StreamEventType.END:
                usage = event.usage.total_tokens if event.usage else "unknown"
                print(
                    f"\n[stream_end chunks={chunk_count} stop={event.stop_reason} tokens={usage}]"
                )


if __name__ == "__main__":
    asyncio.run(main())
