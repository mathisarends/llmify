"""
Test mixing Function and Raw Schema tools together.
"""

import asyncio

from llmify import ChatOpenAI, UserMessage, ToolResultMessage, tool
from llmify.tools import RawSchemaTool


# Function tool
@tool
def get_weather(location: str) -> str:
    """Get weather information for a location"""
    return f"Weather in {location}: 22°C, Sunny"


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression"""
    try:
        result = eval(expression)  # Don't use in production!
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


# Raw schema tool
time_tool = RawSchemaTool(
    name="get_time",
    description="Get the current time for a location",
    schema={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"},
            "timezone": {"type": "string", "description": "Timezone (optional)"},
        },
        "required": ["location"],
    },
)


async def execute_tool(tool_call):
    """Execute any tool call"""
    name = tool_call.name
    args = tool_call.tool

    if name == "get_weather":
        return get_weather(**args)

    elif name == "calculate":
        return calculate(**args)

    elif name == "get_time":
        loc = args.get("location", "Unknown")
        return f"Current time in {loc}: 14:30"

    return f"Unknown tool: {name}"


async def main():
    llm = ChatOpenAI(model="gpt-4o")

    # Mix function and raw schema tools!
    tools = [get_weather, calculate, time_tool]

    messages = [
        UserMessage(
            "What's the weather in Paris? Also, what's 15 * 7, and what time is it in London?"
        )
    ]

    print("🤖 Testing mixed tool types (Function + Raw Schema)\n")
    print("📦 Tools available:")
    for t in tools:
        print(f"   - {t.name}")
    print()

    max_iterations = 3
    for i in range(max_iterations):
        print(f"--- Iteration {i + 1} ---")

        response = await llm.invoke(messages, tools=tools)

        if response.has_tool_calls:
            print(f"🔧 Model called {len(response.tool_calls)} tool(s):")

            messages.append(response.to_message())

            for tc in response.tool_calls:
                print(f"   - {tc.name}({tc.tool})")
                result = await execute_tool(tc)
                print(f"     → {result}")

                messages.append(ToolResultMessage(tool_call_id=tc.id, content=result))
            print()
        else:
            print(f"✅ Final Answer:\n{response.content}\n")
            break
    else:
        print("⚠️ Max iterations reached")


if __name__ == "__main__":
    asyncio.run(main())
