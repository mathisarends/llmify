"""Performance benchmark for cold versus prewarmed Codex Responses WebSockets.

Install ``py-llmify[websocket]`` and authenticate with ``codex login`` first.
The critical-path metric is request-to-first-text-delta: in the cold case it
includes the WebSocket handshake; in the prewarmed case the handshake has
already completed before the request timer starts.
"""

import argparse
import asyncio
import statistics
from dataclasses import dataclass
from time import perf_counter

from llmify import (
    ChatCodex,
    OpenAIResponsesStreamEnd,
    StreamTextDelta,
    UserMessage,
    WebSocketResponsesTransport,
)


@dataclass
class Measurement:
    handshake_ms: float
    first_text_ms: float
    complete_ms: float


async def _request(model: ChatCodex) -> tuple[float, float]:
    started = perf_counter()
    first_text: float | None = None
    completed: float | None = None
    async for event in model.stream(
        [UserMessage(content="Reply with exactly: OK")],
    ):
        if first_text is None and isinstance(event, StreamTextDelta):
            first_text = perf_counter()
        if isinstance(event, OpenAIResponsesStreamEnd):
            completed = perf_counter()

    if first_text is None:
        raise RuntimeError("The response completed without a text delta.")
    if completed is None:
        raise RuntimeError("The stream ended without a terminal response event.")
    return (first_text - started) * 1000, (completed - started) * 1000


async def _cold(model_name: str) -> Measurement:
    model = ChatCodex.from_cli(
        model=model_name,
        transport=WebSocketResponsesTransport(),
    )
    try:
        first_text_ms, complete_ms = await _request(model)
        return Measurement(0.0, first_text_ms, complete_ms)
    finally:
        await model.aclose()


async def _prewarmed(model_name: str) -> Measurement:
    model = ChatCodex.from_cli(
        model=model_name,
        transport=WebSocketResponsesTransport(),
    )
    try:
        started = perf_counter()
        await model.prewarm()
        handshake_ms = (perf_counter() - started) * 1000
        first_text_ms, complete_ms = await _request(model)
        return Measurement(handshake_ms, first_text_ms, complete_ms)
    finally:
        await model.aclose()


def _median(values: list[float]) -> float:
    return statistics.median(values)


async def main(model_name: str, rounds: int) -> None:
    cold: list[Measurement] = []
    warm: list[Measurement] = []

    print(f"Model: {model_name} | rounds per mode: {rounds}")
    for index in range(rounds):
        cold_result = await _cold(model_name)
        warm_result = await _prewarmed(model_name)
        cold.append(cold_result)
        warm.append(warm_result)
        print(
            f"round {index + 1}: "
            f"cold TTFT={cold_result.first_text_ms:.1f} ms | "
            f"prewarm={warm_result.handshake_ms:.1f} ms, "
            f"warm TTFT={warm_result.first_text_ms:.1f} ms"
        )

    cold_ttft = _median([item.first_text_ms for item in cold])
    warm_ttft = _median([item.first_text_ms for item in warm])
    saved = cold_ttft - warm_ttft
    percent = saved / cold_ttft * 100 if cold_ttft else 0.0

    print("\nMedian summary")
    print(f"  cold request -> first text:       {cold_ttft:8.1f} ms")
    print(
        f"  prewarm handshake (off-path):     {_median([item.handshake_ms for item in warm]):8.1f} ms"
    )
    print(f"  prewarmed request -> first text:  {warm_ttft:8.1f} ms")
    print(f"  critical-path difference:         {saved:8.1f} ms ({percent:+.1f}%)")
    print(
        f"  cold/prewarmed completion:        "
        f"{_median([item.complete_ms for item in cold]):.1f} / "
        f"{_median([item.complete_ms for item in warm]):.1f} ms"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.3-codex-spark")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    asyncio.run(main(args.model, args.rounds))
