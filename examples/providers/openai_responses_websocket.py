"""Responses API over WebSocket transport.

Install the optional transport dependency first:
    pip install "py-llmify[websocket]"

Requires OPENAI_API_KEY. ``store=True`` allows the second, separate WebSocket
session to continue from the first response by ID.
"""

import asyncio

from dotenv import load_dotenv

from llmify import (
    ChatOpenAIResponses,
    ContinuationMode,
    OpenAIResponsesStreamEnd,
    ResponsesOptions,
    StreamEventType,
    SystemMessage,
    UserMessage,
    WebSocketResponsesTransport,
)


async def main() -> None:
    load_dotenv(override=True)

    llm = ChatOpenAIResponses(
        model="gpt-5.6",
        store=True,
        transport=WebSocketResponsesTransport(),
        responses_options=ResponsesOptions(
            continuation_mode=ContinuationMode.PREVIOUS_RESPONSE_ID,
            reasoning_summary="concise",
        ),
    )

    first = await llm.invoke(
        [
            SystemMessage(content="You are a concise cooking assistant."),
            UserMessage(content="Give me a simple three-step pasta recipe."),
        ]
    )
    print("First answer:\n", first.completion)
    print("Response ID:", first.provider_state.response_id)

    # This stream opens a new WebSocket connection. Because store=True, the
    # response ID can still be used to continue the server-side conversation.
    print("\nFollow-up (streamed): ", end="", flush=True)
    async for event in llm.stream(
        [UserMessage(content="Make it vegan and include one optional topping.")],
        provider_state=first.provider_state,
    ):
        if event.type == StreamEventType.TEXT:
            print(event.delta, end="", flush=True)
        elif isinstance(event, OpenAIResponsesStreamEnd):
            print("\n\n[complete]")
            print("New response ID:", event.provider_state.response_id)
            print("Reasoning summary:", event.reasoning_summary or "(none)")
            if event.usage:
                print("Total tokens:", event.usage.total_tokens)


if __name__ == "__main__":
    asyncio.run(main())
