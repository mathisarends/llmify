import asyncio

from llmify import ChatOpenAI, tool
from llmify.messages import UserMessage, ToolResultMessage


# ============================================================================
# Tool Definitions
# ============================================================================


@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location"""
    # Mock weather data
    weather_data = {
        "Berlin": {"temp": 15, "condition": "Cloudy"},
        "London": {"temp": 12, "condition": "Rainy"},
        "Tokyo": {"temp": 22, "condition": "Sunny"},
    }

    loc = location.title()
    data = weather_data.get(loc, {"temp": 20, "condition": "Clear"})

    temp = data["temp"]
    if unit == "fahrenheit":
        temp = int(temp * 9 / 5 + 32)

    unit_symbol = "°F" if unit == "fahrenheit" else "°C"
    return f"Weather in {loc}: {temp}{unit_symbol}, {data['condition']}"


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information"""
    return f"Found {max_results} results for '{query}': [Mock search results]"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient"""
    return f"Email sent to {to} with subject '{subject}'"


# ============================================================================
# Main Agent Loop
# ============================================================================


async def run_agent(query: str, model: ChatOpenAI, tools: list):
    """Run agent with tool calling loop"""

    messages = [UserMessage(query)]
    max_iterations = 5

    print(f"🤖 User: {query}\n")

    for iteration in range(max_iterations):
        print(f"--- Iteration {iteration + 1} ---")

        # Get model response
        response = await model.invoke(messages, tools=tools)

        # Check if tools were called
        if response.has_tool_calls:
            print(f"🔧 Model wants to use {len(response.tool_calls)} tool(s):")

            # Add the assistant's message with tool calls to history
            messages.append(response.to_message())

            # Execute each tool call
            for tool_call in response.tool_calls:
                print(f"   - {tool_call.name}({tool_call.tool})")

                # Execute the appropriate tool
                if tool_call.name == "get_weather":
                    result = get_weather(**tool_call.tool)
                elif tool_call.name == "search_web":
                    result = search_web(**tool_call.tool)
                elif tool_call.name == "send_email":
                    result = send_email(**tool_call.tool)
                else:
                    result = "Unknown tool"

                print(f"     Result: {result}")

                # Add tool result to messages
                messages.append(
                    ToolResultMessage(tool_call_id=tool_call.id, content=result)
                )

            print()
        else:
            # No more tool calls, we have the final answer
            print(f"✅ Final Answer: {response.content}\n")
            return response.content

    print("⚠️  Max iterations reached without final answer\n")
    return None


if __name__ == "__main__":

    async def main():
        llm = ChatOpenAI(model="gpt-4o")

        tools = [get_weather, search_web, send_email]

        query = "What's the weather in Berlin? Also, search for nearby Italian restaurants and email me the details."

        await run_agent(query, llm, tools)

    asyncio.run(main())
