import asyncio
import json
from llmify import (
    ChatOpenAI,
    UserMessage,
    ToolResultMessage,
    AssistantMessage,
    RawSchemaTool,
)
from dotenv import load_dotenv

load_dotenv(override=True)


search_tool = RawSchemaTool(
    name="search_web",
    description="Search the web for information",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {
                "type": "integer",
                "description": "Max number of results",
                "default": 5,
            },
        },
        "required": ["query"],
    },
)


def search_web(query: str, max_results: int = 5) -> str:
    return f"Top {max_results} results for '{query}': [Result 1, Result 2, Result 3]"


async def main():
    llm = ChatOpenAI(model="gpt-4o-mini")

    messages = [
        UserMessage(content="Search the web for the latest Python 3.13 features.")
    ]

    print("🤖 Testing RawSchemaTool\n")

    response = await llm.invoke(messages, tools=[search_tool], tool_choice="required")

    if response.tool_calls:
        tc = response.tool_calls[0]
        print(f"✅ Tool called: {tc.function.name}")
        args = json.loads(tc.function.arguments)
        print(f"   Arguments: {args}")

        result = search_web(**args)
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
