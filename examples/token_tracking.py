import asyncio
from llmify import ChatOpenAI, UserMessage
from llmify.tokens import TokenTracker, Trackable, calculate_costs
from dotenv import load_dotenv

load_dotenv(override=True)


async def main():
    llm = ChatOpenAI(model="gpt-4o")

    # `model` is a required field on every provider, exposed via `.model`.
    print(f"Talking to: {llm.model}")

    # Token tracking is opt-in: create a TokenTracker yourself and feed it the
    # usage you care about. The model name comes for free from `llm.model`.
    tracker = TokenTracker()
    history: list[Trackable] = []

    # `add` accepts a full ChatInvokeCompletion directly...
    response = await llm.invoke([UserMessage(content="Write a haiku about Python.")])
    history.append(response)
    tracker.add(response, model=llm.model)

    # ...or a StreamEnd event (or a raw ChatInvokeUsage).
    async for event in llm.stream([UserMessage(content="And one about Rust.")]):
        if event.type == "end":
            history.append(event)
            tracker.add(event, model=llm.model)

    summary = tracker.summary()
    print(f"\ncalls={summary.entry_count} total_tokens={summary.total_tokens}")
    print(
        f"prompt={summary.total_prompt_tokens} "
        f"completion={summary.total_completion_tokens} "
        f"cached={summary.total_prompt_cached_tokens}"
    )

    for entry in tracker.entries:
        print(f"  {entry.model}: {entry.total_tokens} tokens")

    costs = tracker.cost_summary()
    print(f"\nestimated cost: {costs.total_cost:.6f} {costs.currency}")
    assert costs == calculate_costs(history, model=llm.model)


if __name__ == "__main__":
    asyncio.run(main())
