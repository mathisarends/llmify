import asyncio
import json
from llmify import ChatOpenAI, UserMessage, ToolResultMessage, AssistantMessage, tool
from dotenv import load_dotenv

load_dotenv(override=True)


@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    return f"Weather in {location}: 15°{unit[0].upper()}, Cloudy"


async def main():
    llm = ChatOpenAI(model="gpt-4o")

    messages = [UserMessage(content="What's the weather in Tokyo?")]

    print("🤖 Testing function tool with @tool decorator\n")

    response = await llm.invoke(messages, tools=[get_weather], tool_choice="required")

    if response.tool_calls:
        tc = response.tool_calls[0]
        print(f"✅ Tool called: {tc.function.name}")
        args = json.loads(tc.function.arguments)
        print(f"   Arguments: {args}")

        result = get_weather(**args)
        print(f"   Result: {result}")

        messages.append(
            AssistantMessage(
                content=response.completion, tool_calls=response.tool_calls
            )
        )
        messages.append(ToolResultMessage(tool_call_id=tc.id, content=result))

        final = await llm.invoke(messages)
        print(f"\n✅ Final Answer: {final.completion}")
    else:
        print(f"❌ No tool calls: {response.completion}")


if __name__ == "__main__":
    asyncio.run(main())
