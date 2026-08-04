"""Responses API: consume text, reasoning, native output items, and the end event."""

import asyncio

from dotenv import load_dotenv

from llmify import (
    ChatOpenAIResponses,
    OpenAIResponsesStreamEnd,
    ResponsesOptions,
    StreamEventType,
    StreamOutputItemAdded,
    StreamOutputItemDone,
    StreamReasoningSummaryDelta,
    SystemMessage,
    UserMessage,
)


async def main() -> None:
    load_dotenv(override=True)

    llm = ChatOpenAIResponses(
        model="gpt-5.6",
        responses_options=ResponsesOptions(reasoning_summary="concise"),
    )

    print("Answer: ", end="", flush=True)
    async for event in llm.stream(
        [
            SystemMessage(content="Answer clearly and briefly."),
            UserMessage(
                content="Why does a metal spoon feel cold at room temperature?"
            ),
        ]
    ):
        # Generic events remain compatible with the other llmify providers.
        if event.type == StreamEventType.TEXT:
            print(event.delta, end="", flush=True)

        # Responses-specific events give access to provider-native details.
        elif isinstance(event, StreamReasoningSummaryDelta):
            print(f"\n[reasoning summary] {event.delta}", end="", flush=True)
        elif isinstance(event, StreamOutputItemAdded):
            print(
                f"\n[item {event.output_index} started: {event.item['type']}]",
                flush=True,
            )
        elif isinstance(event, StreamOutputItemDone):
            print(
                f"\n[item {event.output_index} finished: {event.item['type']}]",
                flush=True,
            )
        elif isinstance(event, OpenAIResponsesStreamEnd):
            print("\n\n[stream complete]")
            print("Assembled text:", event.completion)
            print("Reasoning summary:", event.reasoning_summary or "(none)")
            print("Response ID:", event.provider_state.response_id)
            print("Native output items:", len(event.provider_state.output_items))
            if event.usage:
                print(
                    "Usage:",
                    f"total={event.usage.total_tokens}",
                    f"reasoning={event.usage.reasoning_tokens}",
                    f"cache-write={event.usage.prompt_cache_write_tokens}",
                )


if __name__ == "__main__":
    asyncio.run(main())
