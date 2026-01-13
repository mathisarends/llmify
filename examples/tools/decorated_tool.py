import asyncio
from llmify import ChatOpenAI, UserMessage, ToolResultMessage, tool


@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    return f"Weather in {location}: 15°{unit[0].upper()}, Cloudy"


async def main():
    llm = ChatOpenAI(model="gpt-4o")

    messages = [UserMessage("What's the weather in Tokyo?")]

    print("🤖 Testing function tool with @tool decorator\n")

    response = await llm.invoke(messages, tools=[get_weather])

    if response.has_tool_calls:
        print(f"✅ Tool called: {response.tool_calls[0].name}")
        print(f"   Arguments: {response.tool_calls[0].tool}")

        result = get_weather(**response.tool_calls[0].tool)
        print(f"   Result: {result}")

        messages.append(response.to_message())
        messages.append(
            ToolResultMessage(tool_call_id=response.tool_calls[0].id, content=result)
        )

        final = await llm.invoke(messages, tools=[get_weather])
        print(f"\n✅ Final Answer: {final.content}")
    else:
        print(f"❌ No tool calls: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
