"""Responses API: continuation, prompt caching, and native completion metadata.

Requires OPENAI_API_KEY. Some models do not support explicit prompt-cache
breakpoints; change ``mode`` to ``"implicit"`` if your model rejects them.
"""

import asyncio

from dotenv import load_dotenv

from llmify import (
    ChatOpenAIResponses,
    ContinuationMode,
    PromptCacheOptions,
    ResponsesOptions,
    SystemMessage,
    UserMessage,
)


async def main() -> None:
    load_dotenv(override=True)

    options = ResponsesOptions(
        # Subsequent calls send only new items when the prior response is stored.
        continuation_mode=ContinuationMode.PREVIOUS_RESPONSE_ID,
        # Request an opaque, provider-managed summary instead of chain-of-thought.
        reasoning_summary="concise",
        prompt_cache_key="examples:responses-api:assistant-v1",
        prompt_cache_options=PromptCacheOptions(mode="explicit", ttl="30m"),
    )
    llm = ChatOpenAIResponses(
        model="gpt-5.6",
        store=True,  # required for previous_response_id across HTTP requests
        responses_options=options,
    )

    first = await llm.invoke(
        [
            SystemMessage(
                content="You are a concise travel assistant.",
                # An explicit cache breakpoint after stable instructions.
                cache=True,
            ),
            UserMessage(content="Plan a one-day visit to Berlin."),
        ]
    )
    print("First answer:\n", first.completion)
    print_completion_metadata(first)

    # Pass the native state and only the new user message.  The provider state is
    # Pydantic-serializable, so it can also be persisted with model_dump_json().
    second = await llm.invoke(
        [UserMessage(content="Adapt it for rainy weather and a 10:00 start.")],
        provider_state=first.provider_state,
    )
    print("\nFollow-up answer:\n", second.completion)
    print_completion_metadata(second)


def print_completion_metadata(response) -> None:
    """Inspect Responses-only completion fields shared by invoke and streams."""
    state = response.provider_state
    print("\nReasoning summary:", response.reasoning_summary or "(none)")
    print("Response ID:", state.response_id)
    print("Native output items:", len(state.output_items))
    print("Local replay window:", len(state.input_items), "items")

    if response.usage:
        print(
            "Usage:",
            f"prompt={response.usage.prompt_tokens}",
            f"completion={response.usage.completion_tokens}",
            f"reasoning={response.usage.reasoning_tokens}",
            f"cache-write={response.usage.prompt_cache_write_tokens}",
        )


if __name__ == "__main__":
    asyncio.run(main())
