import asyncio
import json
import time

from dotenv import load_dotenv

from llmify import (
    AssistantMessage,
    ChatOpenAI,
    StreamEventType,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
    tool,
)


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get weather for a city."""
    return f"The weather in {city} is 22 degrees {unit}."


TOOLS = {"get_weather": get_weather}


async def run_round(
    llm: ChatOpenAI, messages: list, round_idx: int
) -> tuple[str, list]:
    """Stream one round. Returns (assembled_text, collected_tool_calls)."""
    print(f"\n--- Round {round_idx} ---")
    print(
        "streaming live tokens (format: [index +dt ms since previous token] delta):\n"
    )

    assembled = ""
    tool_calls = []
    chunk_count = 0
    last_token_ts = time.perf_counter()
    start_ts = last_token_ts

    async for event in llm.stream(messages, tools=[get_weather]):
        match event.type:
            case StreamEventType.TEXT:
                now = time.perf_counter()
                delta_ms = (now - last_token_ts) * 1000
                last_token_ts = now
                chunk_count += 1
                assembled += event.delta
                # Show every single token with timing — clear proof of streaming.
                print(
                    f"  [{chunk_count:02d} +{delta_ms:6.1f}ms] {event.delta!r}",
                    flush=True,
                )

            case StreamEventType.TOOL_CALL:
                tc = event.tool_call
                args = json.loads(tc.function.arguments)
                fn = TOOLS[tc.function.name]
                result = fn(**args)
                tool_calls.append((tc, result))
                print(
                    f"\n  [TOOL CALL] id={tc.id}"
                    f"\n     name   : {tc.function.name}"
                    f"\n     args   : {args}"
                    f"\n     result : {result}\n"
                )

            case StreamEventType.END:
                total_ms = (time.perf_counter() - start_ts) * 1000
                usage = event.usage.total_tokens if event.usage else "?"
                print(
                    f"\n  [END] "
                    f"stop_reason={event.stop_reason} "
                    f"text_chunks={chunk_count} "
                    f"tokens={usage} "
                    f"elapsed={total_ms:.0f}ms"
                )

    return assembled, tool_calls


async def main() -> None:
    load_dotenv(override=True)

    llm = ChatOpenAI()
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(
            content=(
                "What is the weather in Berlin? Use the get_weather tool, "
                "then answer with a friendly spoken greeting that includes the weather."
            )
        ),
    ]

    print("=" * 72)
    print("Streaming demo: live tokens + tool call round-trip")
    print("=" * 72)

    final_text = ""
    for round_idx in range(1, 4):  # safety cap
        text, tool_calls = await run_round(llm, messages, round_idx)

        if not tool_calls:
            final_text = text
            break

        # Feed the assistant's tool-calling turn + tool results back into the conversation
        # so the next round can produce the spoken answer as streamed text.
        messages.append(
            AssistantMessage(
                content=text or None, tool_calls=[tc for tc, _ in tool_calls]
            )
        )
        for tc, result in tool_calls:
            messages.append(ToolResultMessage(tool_call_id=tc.id, content=result))

    print("\n" + "=" * 72)
    print("Final assembled answer (full text after streaming):")
    print("=" * 72)
    print(final_text)
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
