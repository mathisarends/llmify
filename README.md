# llmify

A lightweight, type-safe Python library for LLM chat completions.

**Features:**
- 🎯 Simple, intuitive API for OpenAI and Azure OpenAI
- 📝 Type-safe structured outputs with Pydantic
- 🛠️ Built-in tool calling support
- 🌊 Async streaming
- 🖼️ Image analysis support
- ⚡ Minimal dependencies, maximum flexibility

## Installation
```bash
pip install py-llmify
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
    # Output: John, 32, data scientist

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
print(response.tool_calls[0].function.name)       # "get_weather"
print(json.loads(response.tool_calls[0].function.arguments))  # {"city": "..."}
```

### Streaming

```python
async def main():
    llm = ChatOpenAI(model="gpt-4o")

    async for chunk in llm.stream([UserMessage(content="Write a haiku about Python")]):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

## Configuration

### Environment Variables
```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://.openai.azure.com/"
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

**Supported Parameters:**
- `temperature` — Creativity (0–2)
- `max_tokens` — Maximum response length
- `top_p` — Nucleus sampling
- `frequency_penalty` — Reduce repetition
- `presence_penalty` — Encourage diversity
- `stop` — Stop sequences
- `seed` — Deterministic outputs

## Providers

### OpenAI
```python
from llmify import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    api_key="sk-..."  # Optional if OPENAI_API_KEY is set
)
```

### Azure OpenAI
```python
from llmify import ChatAzureOpenAI

llm = ChatAzureOpenAI(
    model="gpt-4o",
    api_key="...",              # Optional if AZURE_OPENAI_API_KEY is set
    azure_endpoint="https://.openai.azure.com/",  # Optional if env var is set
)
```

## Design Philosophy

**Lightweight & Focused**
- Thin wrapper around official SDKs
- Minimal dependencies
- No unnecessary abstractions

**Type-Safe & Modern**
- Full type hints throughout
- Pydantic for all messages and responses
- Async-first design

## License

MIT
