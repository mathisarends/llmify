# llmify

![llmify banner](static/banner.png)

A type-safe Python library for LLM chat completions.

**Features:**

- Simple, intuitive API for OpenAI, Codex, Azure OpenAI, Cerebras, Anthropic, and Google Gemini
- Type-safe structured outputs with Pydantic
- Built-in tool calling support
- Async streaming
- Image analysis support
- Automatic retries for transient failures, with per-retry callbacks
- Optional token usage and cost tracking

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Features](#core-features)
  - [Message Types](#message-types)
  - [Structured Outputs](#structured-outputs)
  - [Tool Calling](#tool-calling)
  - [Streaming](#streaming)
  - [Retries](#retries)
  - [Token Usage Tracking](#token-usage-tracking)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Model Parameters](#model-parameters)
- [Providers](#providers)
  - [OpenAI](#openai)
  - [OpenAI Responses API](#openai-responses-api)
  - [Codex](#codex)
  - [Azure OpenAI](#azure-openai)
  - [Anthropic](#anthropic)
  - [Cerebras](#cerebras)
  - [Google Gemini](#google-gemini)
- [Credits](#credits)
- [License](#license)

## Installation

```bash
pip install py-llmify
```

Install only the provider you need:

```bash
pip install py-llmify[openai]      # OpenAI + Azure OpenAI
pip install py-llmify[cerebras]    # Cerebras
pip install py-llmify[anthropic]   # Anthropic (Claude)
pip install py-llmify[google]      # Google Gemini
pip install py-llmify[all]         # All providers
```

Extras can be combined, for example:

```bash
pip install py-llmify[openai,google]
```

## Quick Start

```python
import asyncio
from llmify import ChatOpenAI, UserMessage, SystemMessage

async def main():
    llm = ChatOpenAI(model="gpt-4o")

    response = await llm.invoke([
        SystemMessage(content="You are a helpful assistant"),
        UserMessage(content="What is 2+2?")
    ])

    print(response.completion)  # "2+2 equals 4"

asyncio.run(main())
```

All `invoke` calls return a `ChatInvokeCompletion[T]` with:

- `completion` — the text (or parsed Pydantic model) returned by the model
- `tool_calls` — list of `ToolCall` objects, if any
- `usage` — token usage (`ChatInvokeUsage`)
- `stop_reason` — why the model stopped

## Core Features

### Message Types

```python
from llmify import SystemMessage, UserMessage, AssistantMessage, ToolResultMessage

messages = [
    SystemMessage(content="You are a Python expert"),
    UserMessage(content="How do I read a file?"),
    AssistantMessage(content="You can use open() with a context manager"),
    UserMessage(content="Show me an example"),
]
```

#### Image messages

Pass images inline inside a `UserMessage` using content parts:

```python
from llmify import UserMessage, ContentPartTextParam, ContentPartImageParam, ImageURL

message = UserMessage(
    content=[
        ContentPartTextParam(text="What's in this image?"),
        ContentPartImageParam(
            image_url=ImageURL(
                url="data:image/jpeg;base64,<base64data>",
                media_type="image/jpeg",
                detail="high",
            )
        ),
    ]
)
```

### Structured Outputs

Pass `output_format` to get a validated Pydantic model back:

```python
from pydantic import BaseModel
from llmify import ChatOpenAI, UserMessage

class Person(BaseModel):
    name: str
    age: int
    occupation: str

async def main():
    llm = ChatOpenAI(model="gpt-4o")

    response = await llm.invoke(
        [UserMessage(content="Extract: John is 32 and works as a data scientist")],
        output_format=Person,
    )

    person = response.completion  # type: Person
    print(f"{person.name}, {person.age}, {person.occupation}")
    # John, 32, data scientist

asyncio.run(main())
```

### Tool Calling

#### `@tool` decorator

Define tools from plain Python functions:

```python
import json
from llmify import ChatOpenAI, UserMessage, AssistantMessage, ToolResultMessage, tool

@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get current weather for a location"""
    return f"Weather in {location}: 22°{unit[0].upper()}, Sunny"

async def main():
    llm = ChatOpenAI(model="gpt-4o")
    messages = [UserMessage(content="What's the weather in Paris?")]

    response = await llm.invoke(messages, tools=[get_weather])

    if response.tool_calls:
        tc = response.tool_calls[0]
        args = json.loads(tc.function.arguments)
        result = get_weather(**args)

        messages.append(AssistantMessage(content=response.completion, tool_calls=response.tool_calls))
        messages.append(ToolResultMessage(tool_call_id=tc.id, content=result))

        final = await llm.invoke(messages)
        print(final.completion)

asyncio.run(main())
```

#### `RawSchemaTool`

Use a raw JSON schema when you need full control over the tool definition:

```python
import json
from llmify import ChatOpenAI, UserMessage, AssistantMessage, ToolResultMessage, RawSchemaTool

search_tool = RawSchemaTool(
    name="search_web",
    description="Search the web for information",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    },
)

async def main():
    llm = ChatOpenAI(model="gpt-4o-mini")
    messages = [UserMessage(content="Search for Python 3.13 features")]

    response = await llm.invoke(messages, tools=[search_tool])

    if response.tool_calls:
        tc = response.tool_calls[0]
        args = json.loads(tc.function.arguments)
        result = my_search_fn(**args)

        messages.append(AssistantMessage(content=response.completion, tool_calls=response.tool_calls))
        messages.append(ToolResultMessage(tool_call_id=tc.id, content=result))

        final = await llm.invoke(messages)
        print(final.completion)

asyncio.run(main())
```

#### Dict schema

Pass raw OpenAI-style tool dicts directly:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    }
]

response = await llm.invoke(messages, tools=tools)
print(response.tool_calls[0].function.name)
print(json.loads(response.tool_calls[0].function.arguments))
```

### Streaming

```python
import json
from llmify import ChatOpenAI, UserMessage, StreamEventType

async def main():
    llm = ChatOpenAI()
    chunk_count = 0

    async for event in llm.stream([UserMessage(content="Write a haiku about Python")]):
        if event.type is StreamEventType.TEXT:
            chunk_count += 1
            print(f"[{chunk_count:02d}]{event.delta}", end="", flush=True)
        elif event.type is StreamEventType.END:
            print(f"\n[stream_end stop={event.stop_reason}]")

asyncio.run(main())
```

For streaming with tools, handle `StreamEventType.TOOL_CALL` and parse the complete JSON arguments:

```python
import json
from llmify import ChatOpenAI, UserMessage, StreamEventType

async def main():
    llm = ChatOpenAI()

    async for event in llm.stream(messages, tools=[get_weather]):
        if event.type is StreamEventType.TEXT:
            print(event.delta, end="", flush=True)
        elif event.type is StreamEventType.TOOL_CALL:
            args = json.loads(event.tool_call.function.arguments)
            result = get_weather(**args)
            print(f"\n[tool_result] {result}")
        elif event.type is StreamEventType.END:
            print(f"\n[stream_end stop={event.stop_reason} tokens={event.usage.total_tokens if event.usage else 'unknown'}]")

asyncio.run(main())
```

Full runnable example: `examples/streaming_tool_calls.py`

### Retries

All bundled providers retry transient connection, timeout, rate-limit, and server
errors through the same llmify retry layer. `max_retries` is the number of
additional attempts after the initial request and defaults to `2`; set it to `0`
to disable automatic retries:

```python
llm = ChatOpenAIResponses(model="gpt-5.4-mini", max_retries=5)
```

Rate-limit `Retry-After` headers are respected, with exponential backoff and
jitter for other transient failures. `invoke()` safely discards an incomplete
attempt before retrying. `stream()` retries only until its first event has been
emitted; after that it raises `RetryableError` rather than replaying duplicate
output.

Each scheduled retry is reported through a sync or async `on_retry` callback:

```python
from llmify import RetryEvent

def report_retry(event: RetryEvent) -> None:
    print(
        f"Attempt {event.failed_attempt}/{event.max_attempts} failed; "
        f"retry {event.retry_number}/{event.max_retries} "
        f"in {event.delay:.1f}s: {event.error}"
    )

llm = ChatOpenAIResponses(model="gpt-5.4-mini", max_retries=5, on_retry=report_retry)
```

Pass `on_retry` to `invoke()` or `stream()` to override the client-level callback
for a single call. Callback exceptions cancel the retry and propagate to the caller.

### Token Usage Tracking

Every response carries `usage`, and every provider exposes its model as `llm.model`.

```python
response = await llm.invoke([UserMessage(content="Hi")])
print(response.usage)
```

## Configuration

### Environment Variables

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Codex
export CODEX_ACCESS_KEY="..."
export CODEX_ACCOUNT_ID="..."

# Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"

# Cerebras
export CEREBRAS_API_KEY="csk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GEMINI_API_KEY="..."
```

### Model Parameters

Set defaults when initializing or override per request:

```python
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
)

response = await llm.invoke(
    messages=[UserMessage(content="Hi")],
    temperature=0.2,
    max_tokens=500,
)
```

Supported parameters: `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, `stop`, `seed`.

## Providers

### OpenAI

```python
from llmify import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    api_key="sk-...",  # optional if OPENAI_API_KEY is set
    base_url="https://...",  # optional, defaults to the OpenAI API
    default_headers={"X-My-Header": "value"},  # optional
)
```

`api_key` also accepts an async callable (`() -> str`), which is awaited before every
request — useful for short-lived tokens that need refreshing.

### OpenAI Responses API

```python
from llmify import ChatOpenAIResponses

llm = ChatOpenAIResponses(
    model="gpt-5.4-mini",
    api_key="sk-...",  # optional if OPENAI_API_KEY is set
    base_url="https://...",  # optional, defaults to the OpenAI API
)
```

Use `ChatOpenAIResponses` when an endpoint exposes OpenAI's Responses API rather
than the Chat Completions API. It supports the same llmify `invoke` and `stream`
interface.

For reasoning models, `reasoning_effort` sets how much the model thinks before
answering — `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"` or `"xhigh"`:

```python
llm = ChatOpenAIResponses(model="gpt-5.4-mini", reasoning_effort="high")

# per call, overriding the default above
await llm.invoke(messages, reasoning_effort="low")
```

Which levels a model accepts differs — `"xhigh"` is limited to the newest
reasoning models — and an unsupported level comes back as a request error.

#### Native Responses state

Responses calls return an `OpenAIResponsesCompletion` with an explicit,
serializable `provider_state`. The state contains the response ID, the complete
local replay window, and every native `response.output_item.done` item (including
reasoning, messages, and function calls):

```python
from llmify import ChatOpenAIResponses, UserMessage

llm = ChatOpenAIResponses(model="gpt-5.6", store=False)

first = await llm.invoke([UserMessage(content="Inspect this problem")])
second = await llm.invoke(
    [UserMessage(content="Now refine the answer")],  # only new input
    provider_state=first.provider_state,
)
```

Stateless mode is the default. With `store=False`, encrypted reasoning is
requested and replayed unchanged; it is opaque provider state, not readable
chain-of-thought. Use `ContinuationMode.PREVIOUS_RESPONSE_ID` to send only new
items when the previous response is available server-side. Instructions are
retained locally and resent because `previous_response_id` does not carry them
forward automatically.

```python
from llmify import ContinuationMode, ResponsesOptions

llm = ChatOpenAIResponses(
    model="gpt-5.6",
    store=True,
    responses_options=ResponsesOptions(
        continuation_mode=ContinuationMode.PREVIOUS_RESPONSE_ID,
    ),
)
```

#### Complete local tool loop

`invoke_with_tools` executes all function calls in a response, feeds every
`function_call_output` back to the model, and repeats until a final answer is
produced. FunctionTool exceptions become structured tool outputs so the model
can recover. `max_tool_rounds` bounds the loop. Dict schemas and
`RawSchemaTool` values need a `tool_executor` callback because they contain no
implementation.

```python
from llmify import UserMessage, tool

@tool
def lookup(query: str) -> str:
    return f"result for {query}"

result = await llm.invoke_with_tools(
    [UserMessage(content="Look up alpha and beta, then compare them")],
    tools=[lookup],
    max_tool_rounds=8,
)
```

#### Reasoning summaries and native stream events

Set `reasoning_summary="auto"`, `"concise"`, or `"detailed"`. Summaries arrive
as `StreamReasoningSummaryDelta` and are never mixed into `StreamTextDelta`.
Responses streams also expose `StreamOutputItemAdded` and
`StreamOutputItemDone`; the final `OpenAIResponsesStreamEnd` always carries the
assembled provider state. These Responses-only events extend the neutral
`StreamProviderEvent` hook rather than changing other providers' event models.

Usage is returned as `OpenAIResponsesUsage`, adding `reasoning_tokens` and
`prompt_cache_write_tokens` to the common token fields.

#### Prompt caching

Use a stable `prompt_cache_key`; keep instructions and tool definitions stable
and ordered. On models supporting explicit breakpoints, `cache=True` marks the
end of a message as reusable provider input:

```python
from llmify import PromptCacheOptions, ResponsesOptions, SystemMessage

options = ResponsesOptions(
    prompt_cache_key="tenant:acme:agent-v1",
    prompt_cache_options=PromptCacheOptions(mode="explicit", ttl="30m"),
)
llm = ChatOpenAIResponses(model="gpt-5.6", responses_options=options)
messages = [SystemMessage(content=large_stable_instructions, cache=True)]
```

Explicit cache options and breakpoints are model-dependent; older models can
reject them. Automatic prompt caching remains available without these options.

#### WebSocket transport

Install the optional transport dependency and select it explicitly:

```console
pip install "py-llmify[websocket]"
```

```python
from llmify import ResponsesOptions, WebSocketResponsesTransport

llm = ChatOpenAIResponses(
    model="gpt-5.6",
    transport=WebSocketResponsesTransport(),
    responses_options=ResponsesOptions(
        continuation_mode="previous_response_id",
    ),
)
```

HTTP/SSE remains the default. A WebSocket `invoke_with_tools` call keeps one
connection open across all model/tool rounds and sends incremental tool outputs
with `previous_response_id`. A standalone WebSocket `invoke` opens one scoped
connection; when `store=False`, a later standalone invocation safely falls back
to the state's full local replay window because connection-local state no longer
exists.

Transport is a port, not a mode flag. `HTTPResponsesTransport` is the default,
`WebSocketResponsesTransport` is opt-in, and custom implementations can provide
the `ResponsesTransport`/`ResponsesSession` protocols for testing or alternate
wire transports. Continuation knowledge remains scoped to the session that owns
it.

### Codex

```python
from llmify import ChatCodex

llm = ChatCodex(
    model="gpt-5.6-terra",
    api_key="...",  # optional if CODEX_ACCESS_KEY is set
    chatgpt_account_id="...",
    reasoning_effort="high",  # optional
)
```

`ChatCodex` specializes `ChatOpenAIResponses` for the Codex endpoint and
configures the required `ChatGPT-Account-Id` header from `chatgpt_account_id`.
The endpoint URL is fixed by the provider and does not need to be supplied by
callers.

This is a reverse-engineered endpoint: it authenticates with a ChatGPT
subscription rather than an API key, and OpenAI does not document or support it.

#### Borrowing the Codex CLI login

If the [Codex CLI](https://github.com/openai/codex) is installed and logged in
(`codex login`), its session can be used directly — no environment variables:

```python
llm = ChatCodex.from_cli(model="gpt-5.6-terra", reasoning_effort="high")
```

`from_cli` takes the same model options as the constructor — only `api_key` and
`chatgpt_account_id` come from the login instead.

This reads `~/.codex/auth.json` (or `$CODEX_HOME/auth.json`) for the account id
and access token — no network access, no writes. From the request path onwards
the token is refreshed as it approaches expiry, and the rotated tokens are
written back so the CLI keeps working. The approach is borrowed from
[llm-openai-via-codex](https://github.com/simonw/llm-openai-via-codex).

For the credentials themselves, a different `auth.json`, or one token provider
shared across several clients, compose the two pieces yourself:

```python
from llmify import ChatCodex, CodexCliAuth
from llmify.auth import read_codex_credentials

credentials = read_codex_credentials()  # or read_codex_credentials(auth_path=...)
print(credentials.expires_in)           # seconds until the access token expires

auth = CodexCliAuth(credentials)
llm = ChatCodex(
    model="gpt-5.6-terra",
    api_key=auth,                       # awaited before every request
    chatgpt_account_id=auth.account_id,
)
```

`read_codex_credentials()` only ever reads the file. Its async counterpart
`refresh_codex_credentials()` is what performs the OAuth refresh and the
write-back — `CodexCliAuth` calls it from the request path when the token is
about to expire, and applications that want to control that themselves can call
it directly.

A missing or unusable login raises `CodexCredentialsError`, a subclass of
`CredentialsUnavailableError`.

Full runnable examples: `examples/providers/borrowed_codex.py` and
`examples/providers/codex_cli_auth.py`

### Azure OpenAI

```python
from llmify import ChatAzureOpenAI

llm = ChatAzureOpenAI(
    model="gpt-4o",
    api_key="...",           # optional if AZURE_OPENAI_API_KEY is set
    azure_endpoint="https://<resource>.openai.azure.com/",  # optional if env var is set
)
```

For Azure's Responses API, use `ChatAzureOpenAIResponses`:

```python
from llmify import ChatAzureOpenAIResponses

llm = ChatAzureOpenAIResponses(
    model="my-gpt-deployment",
    api_key="...",           # optional if AZURE_OPENAI_API_KEY is set
    azure_endpoint="https://<resource>.openai.azure.com/",  # optional if env var is set
    reasoning_effort="high",  # optional
)
```

It provides the same `invoke`, `stream`, structured-output, and tool-calling
interface as `ChatOpenAIResponses` and uses Azure's `/openai/v1/` endpoint.

### Anthropic

```python
from llmify import ChatAnthropic

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    api_key="sk-ant-...",  # optional if ANTHROPIC_API_KEY is set
)
```

The Anthropic provider supports the same API surface — `invoke`, `stream`, structured output, and tool calling — all mapped to the Anthropic messages API under the hood.

### Cerebras

```python
from llmify import ChatCerebras

llm = ChatCerebras(
    model="gpt-oss-120b",
    api_key="csk-...",  # optional if CEREBRAS_API_KEY is set
)
```

The Cerebras provider uses Cerebras' OpenAI-compatible API and supports `invoke`, `stream`, structured output, and tool calling.

### Google Gemini

```python
from llmify import ChatGoogle

llm = ChatGoogle(
    model="gemini-3.5-flash",
    api_key="...",  # optional if GEMINI_API_KEY is set
)
```

The Google provider supports the same API surface: `invoke`, `stream`, structured output, and tool calling.

## Credits

Inspired by [LangChain](https://github.com/langchain-ai/langchain) and [browser-use](https://github.com/browser-use/browser-use).

## License

MIT
