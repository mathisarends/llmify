# Streaming with Tool Calls

## Goal

Extend `ChatModel.stream()` so it supports tool calls. Behavior:

- **Plain text** is streamed token-by-token (as today).
- **Tool calls** are NOT streamed argument-by-argument. Each tool call is buffered internally until its JSON arguments are complete, then emitted as a single, fully-parsed `ToolCall` event.

This keeps the simple "print tokens as they arrive" UX for normal answers, while making tool-call streaming safe to consume (callers never see half-built JSON).

## Current State

`llmify/base.py` defines:

```python
async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[str]: ...
```

Both providers only yield text deltas:

- `llmify/providers/openai_compatible.py:212-225` — iterates `ChatCompletionChunk`s and yields `delta.content`.
- `llmify/providers/anthropic.py:320-326` — uses `client.messages.stream()` and yields from `stream.text_stream`.

Neither accepts `tools`, neither emits tool calls, neither surfaces stop reason or usage.

## API Design

### New event types (`llmify/views.py`)

A discriminated union over what the caller cares about during a stream:

```python
class StreamTextDelta(BaseModel):
    type: Literal["text"] = "text"
    delta: str

class StreamToolCall(BaseModel):
    """Emitted once a tool call's arguments JSON is fully assembled."""
    type: Literal["tool_call"] = "tool_call"
    tool_call: ToolCall

class StreamEnd(BaseModel):
    """Final event. Always emitted exactly once at the end of the stream."""
    type: Literal["end"] = "end"
    stop_reason: str | None = None
    usage: ChatInvokeUsage | None = None
    tool_calls: list[ToolCall] = []   # full list, also useful for callers that only want the end
    completion: str = ""              # full accumulated text

StreamEvent = StreamTextDelta | StreamToolCall | StreamEnd
```

Rationale:
- `StreamTextDelta.delta` mirrors the current `str` payload — easy migration.
- `StreamToolCall` is only emitted when the call is **complete and parseable**. The caller can execute it immediately.
- `StreamEnd` carries finalized metadata (`usage`, `stop_reason`) and a snapshot of the assistant turn, so callers that only need the end can ignore the intermediate events.

### `ChatModel.stream()` signature

Change `stream()` to:

```python
async def stream(
    self,
    messages: list[Message],
    tools: list[Tool | dict] | None = None,
    tool_choice: Literal["auto", "required", "none"] = "auto",
    **kwargs: Any,
) -> AsyncIterator[StreamEvent]: ...
```

This is a **breaking change** to the existing `AsyncIterator[str]` return type. Acceptable here because the library is pre-1.0 and `stream()` has very few external touchpoints. We will update the examples in the same PR. (If we later decide it's too disruptive, the fallback is to keep `stream()` text-only and add a separate `stream_events()` — but for now we go with one method.)

### Iteration ordering guarantees

For any single stream:
1. Zero or more `StreamTextDelta` events, in arrival order.
2. Zero or more `StreamToolCall` events, each emitted as soon as that specific tool call's JSON is complete (so callers can start executing tool A while tool B is still streaming).
3. Exactly one `StreamEnd`.

Text and tool-call events can interleave if the provider produces them that way (OpenAI usually emits text first then tools, Anthropic can interleave text/tool blocks — both are fine here).

## Provider Implementations

### OpenAI / OpenAICompatible (`llmify/providers/openai_compatible.py`)

OpenAI streams tool calls as `delta.tool_calls`, a list of objects with `index`, `id?`, `function.name?`, `function.arguments?`. Each field can arrive across many chunks; only `index` is reliable as the key.

Implementation outline for `stream()`:

```python
openai_tools = [t if isinstance(t, dict) else t.to_openai_schema() for t in tools] if tools else None

stream = await self._client.chat.completions.create(
    model=self._model,
    messages=self._convert_messages(messages),
    tools=openai_tools,
    tool_choice=tool_choice if openai_tools else None,
    stream=True,
    stream_options={"include_usage": True},  # needed to get usage on the last chunk
    **params,
)

# State accumulated across chunks
buffers: dict[int, dict] = {}  # index -> {"id": str, "name": str, "arguments": str, "emitted": bool}
text_acc: list[str] = []
stop_reason: str | None = None
usage: ChatInvokeUsage | None = None

async for chunk in stream:
    if chunk.usage is not None:
        usage = self._parse_usage(chunk.usage)
    if not chunk.choices:
        continue
    choice = chunk.choices[0]
    delta = choice.delta

    if delta.content:
        text_acc.append(delta.content)
        yield StreamTextDelta(delta=delta.content)

    for tc_delta in (delta.tool_calls or []):
        buf = buffers.setdefault(tc_delta.index, {"id": "", "name": "", "arguments": "", "emitted": False})
        if tc_delta.id:           buf["id"] = tc_delta.id
        if tc_delta.function:
            if tc_delta.function.name:      buf["name"] = tc_delta.function.name
            if tc_delta.function.arguments: buf["arguments"] += tc_delta.function.arguments

    if choice.finish_reason:
        stop_reason = choice.finish_reason
        # finish_reason on this choice means: no more deltas for any tool call on this choice → flush
        for idx in sorted(buffers):
            buf = buffers[idx]
            if not buf["emitted"]:
                tc = ToolCall(id=buf["id"], function=Function(name=buf["name"], arguments=buf["arguments"]))
                buf["emitted"] = True
                yield StreamToolCall(tool_call=tc)

yield StreamEnd(
    stop_reason=stop_reason,
    usage=usage,
    tool_calls=[ToolCall(id=b["id"], function=Function(name=b["name"], arguments=b["arguments"]))
                for b in (buffers[i] for i in sorted(buffers))],
    completion="".join(text_acc),
)
```

Notes:
- **`stream_options={"include_usage": True}`** is required to get token usage in stream mode. Without it `chunk.usage` is always `None`.
- We emit tool calls only at `finish_reason`, not earlier. OpenAI does not give a per-tool-call "done" signal mid-stream, and partial JSON is not guaranteed to be parseable. The user's requirement ("warte auf das vollständige JSON") is exactly this.
- `tool_choice` is only passed when tools are present (matches non-streaming path).

### Anthropic (`llmify/providers/anthropic.py`)

Anthropic's stream emits explicit content-block boundaries, which makes tool calls easier than OpenAI:

- `message_start` — start of message, contains initial `usage`.
- `content_block_start { content_block: { type: "tool_use", id, name, input: {} } }`
- `content_block_delta { delta: { type: "input_json_delta", partial_json: "..." } }` — repeated.
- `content_block_stop` — fires when the block is complete. This is the moment to emit `StreamToolCall`.
- `content_block_delta { delta: { type: "text_delta", text: "..." } }` — for text blocks.
- `message_delta` — contains final `stop_reason` and updated `usage` (output tokens).
- `message_stop`.

Use the lower-level event stream (not `text_stream`) so we see tool-use events:

```python
params = self._build_params(messages, kwargs)
anthropic_tools = [t if isinstance(t, dict) else self._convert_tool(t) for t in tools] if tools else None
if anthropic_tools:
    params["tools"] = anthropic_tools
    params["tool_choice"] = {"auto": {"type": "auto"}, "required": {"type": "any"}, "none": {"type": "none"}}[tool_choice]

# Per-block state, keyed by content-block index
blocks: dict[int, dict] = {}     # index -> {"type": "tool_use"|"text", "id", "name", "json": str}
text_acc: list[str] = []
input_tokens = 0
output_tokens = 0
cache_read = None
cache_create = None
stop_reason: str | None = None

async with self._client.messages.stream(**params) as stream:
    async for event in stream:
        if event.type == "message_start":
            u = event.message.usage
            input_tokens = u.input_tokens
            cache_read = getattr(u, "cache_read_input_tokens", None)
            cache_create = getattr(u, "cache_creation_input_tokens", None)

        elif event.type == "content_block_start":
            cb = event.content_block
            if cb.type == "tool_use":
                blocks[event.index] = {"type": "tool_use", "id": cb.id, "name": cb.name, "json": ""}
            elif cb.type == "text":
                blocks[event.index] = {"type": "text"}

        elif event.type == "content_block_delta":
            d = event.delta
            if d.type == "text_delta":
                text_acc.append(d.text)
                yield StreamTextDelta(delta=d.text)
            elif d.type == "input_json_delta":
                blocks[event.index]["json"] += d.partial_json

        elif event.type == "content_block_stop":
            b = blocks.get(event.index)
            if b and b["type"] == "tool_use":
                # block is complete → emit
                yield StreamToolCall(tool_call=ToolCall(
                    id=b["id"],
                    function=Function(name=b["name"], arguments=b["json"] or "{}"),
                ))

        elif event.type == "message_delta":
            stop_reason = event.delta.stop_reason or stop_reason
            output_tokens = event.usage.output_tokens

usage = ChatInvokeUsage(
    prompt_tokens=input_tokens,
    completion_tokens=output_tokens,
    total_tokens=input_tokens + output_tokens,
    prompt_cached_tokens=cache_read,
    prompt_cache_creation_tokens=cache_create,
)

tool_calls = [
    ToolCall(id=b["id"], function=Function(name=b["name"], arguments=b["json"] or "{}"))
    for b in blocks.values() if b["type"] == "tool_use"
]

yield StreamEnd(
    stop_reason=stop_reason,
    usage=usage,
    tool_calls=tool_calls,
    completion="".join(text_acc),
)
```

Notes:
- Anthropic emits `partial_json` strings concatenated to form the full arguments JSON. We append them verbatim and emit only at `content_block_stop`. This satisfies the "wait for complete JSON" rule.
- We store arguments as a **string** (`b["json"]`) to match the existing `Function.arguments: str` contract. We do **not** `json.loads` here — callers can parse if they want (mirroring the non-streaming Anthropic path that does `json.dumps(block.input)` to convert back to a string).
- If a tool_use block has empty input, Anthropic sends no `input_json_delta`. We default to `"{}"` so callers can always `json.loads(arguments)`.

## Files to Change

| File | Change |
|------|--------|
| `llmify/views.py` | Add `StreamTextDelta`, `StreamToolCall`, `StreamEnd`, `StreamEvent` union. |
| `llmify/__init__.py` | Re-export the new types. |
| `llmify/base.py` | Update `stream()` abstract signature: accept `tools`/`tool_choice`, return `AsyncIterator[StreamEvent]`. |
| `llmify/providers/openai_compatible.py` | Rewrite `stream()` per OpenAI section above. |
| `llmify/providers/anthropic.py` | Rewrite `stream()` per Anthropic section above. |
| `examples/` | Add `examples/streaming_tool_calls.py` showing the new pattern. Update any existing streaming examples that consumed `str` directly. |
| `tests/` (if present) | Add tests — see below. |

## Examples to Add

`examples/streaming_tool_calls.py` — demonstrates the canonical loop:

```python
async for event in llm.stream(messages, tools=[get_weather]):
    match event.type:
        case "text":
            print(event.delta, end="", flush=True)
        case "tool_call":
            tc = event.tool_call
            result = get_weather(**json.loads(tc.function.arguments))
            # ...append AssistantMessage + ToolResultMessage, then recurse if desired
        case "end":
            print(f"\n[stop={event.stop_reason}, tokens={event.usage.total_tokens}]")
```

## Edge Cases & Decisions

1. **Partial JSON visibility.** We never expose partial argument strings. Per the user's requirement, the caller waits until the JSON is complete. Cost: a "thinking…" indicator for tool calls must be driven by the absence of text deltas, not by tool-call progress.
2. **Network error mid-stream.** Whatever the SDK raises will propagate out of the async generator. We do **not** emit a partial `StreamEnd`. Callers should wrap iteration in try/except.
3. **Empty tool input.** Defaulted to `"{}"` (Anthropic side). OpenAI always sends at least `arguments: ""` even for no-arg calls; we treat empty string as `"{}"` when constructing the resulting `ToolCall`? → No: keep it as the raw string the model produced, to match the non-streaming path's behavior. Document this.
4. **`tool_choice` without `tools`.** Ignored (matches non-streaming path).
5. **Structured output (`output_format`) in stream.** Out of scope for this plan. `stream()` will not accept `output_format`; users wanting structured output should keep using `invoke()`.
6. **Usage on OpenAI streams.** Requires `stream_options={"include_usage": True}`. Always set when streaming.
7. **Multiple choices (`n > 1`).** Not supported in streaming. We only read `choices[0]`. Matches current `invoke()` behavior.

## Test Plan

Add (or extend) tests covering:

- **Plain text stream** — sequence is `[TextDelta, TextDelta, ..., End]`; concatenated deltas equal `End.completion`.
- **Tool-only stream** — sequence contains one `ToolCall` event per tool, `End.tool_calls` matches the emitted ones, `End.completion == ""`.
- **Mixed stream** — some text deltas followed by a tool call (OpenAI typical) and the reverse interleaving (Anthropic possible).
- **Multi-tool stream** — two parallel tool calls, both fully assembled, emitted in stable order.
- **JSON completeness invariant** — every emitted `StreamToolCall.tool_call.function.arguments` is valid JSON (`json.loads` round-trips).
- **Usage present** — `End.usage` is non-null for both providers.
- **`tool_choice="required"`** — model produces at least one tool call.

Use SDK mocks / VCR-style cassettes if available; otherwise gate behind an integration-test marker that requires API keys.

## Rollout

1. Land event types + base signature change (compiles, breaks providers).
2. Implement OpenAI streaming.
3. Implement Anthropic streaming.
4. Update examples.
5. Add tests.
6. Note the breaking change in the next release's changelog: `stream()` now yields `StreamEvent`, not `str`. Migration is a `match event.type` on the type field, plus passing `tools=` if needed.
