# Terra

Terra is an LLM agent framework for Elixir, built on `:gen_statem`. It provides
a behaviour-based approach to building stateful, streaming agents with tool
execution, context aging, and multi-agent coordination — all on top of the BEAM's
concurrency and fault-tolerance primitives.

## Features

- **State-machine agents** — model agent behaviour explicitly with `:gen_statem`
  states, timeouts, and actions.
- **Streaming, provider-agnostic** — Anthropic, OpenAI, and Google all normalize
  to a single Terra event protocol, so agents work identically across providers.
- **Eager tool execution** — tools run during streaming; results land in your
  `handle_response/2` callback.
- **Context aging** — tool results automatically expire and prune based on
  distance from the most recent turn, keeping the context window lean.
- **Prompt caching** — first-class `Terra.Document` support for large, stable
  content with ephemeral cache control.
- **Multi-agent sessions** — coordinate multiple agents through a shared
  `Terra.Kernel` with locking, metadata, and pub/sub notifications.

## Installation

Add Terra to your `mix.exs` dependencies as a git dependency:

```elixir
defp deps do
  [
    {:terra, github: "anuvaya/terra"}
    # Pin to a tag or ref for reproducible builds:
    # {:terra, github: "anuvaya/terra", tag: "v0.1.0"}
  ]
end
```

## Quick Start

A Terra agent implements five callbacks: `init/1`, `context/2`, `handle_input/3`,
`handle_response/2`, and `handle_stream_event/3`.

```elixir
defmodule MyAgent do
  use Terra.Agent

  def init(_args) do
    {:ok, :idle, %{history: []},
     provider: {Terra.Provider.Anthropic, %{}},
     registries: []}
  end

  def context(_state_name, state) do
    Terra.Context.new()
    |> Terra.Context.system("You are a helpful assistant.")
    |> Terra.Context.history(state.data.history)
    |> Terra.Context.model(%{
      model: "claude-sonnet-4-5-20250929",
      max_tokens: 4096,
      config: %{api_key: System.get_env("ANTHROPIC_API_KEY")}
    })
    |> Terra.Context.age_tools(state)
    |> Terra.Context.build()
  end

  def handle_input(:idle, {:message, text}, state) do
    history = state.data.history ++ [%{role: "user", content: text}]
    {:next_state, :thinking, %{state | data: %{state.data | history: history}}, [:invoke]}
  end

  def handle_stream_event(_state, _event, state), do: {:keep_state, state}

  def handle_response(resp, state) do
    text = Enum.find_value(resp.content_blocks, "", fn
      %{type: "text", text: t} -> t
      _ -> nil
    end)
    history = state.data.history ++ [%{role: "assistant", content: text}]
    {:next_state, :idle, %{state | data: %{state.data | history: history}}}
  end
end
```

Start and interact with the agent:

```elixir
{:ok, pid} = MyAgent.start_link(%{})
Terra.Agent.send_input(pid, {:message, "Hello!"})
```

## Core Concepts

### Tools

Define tools with the `Terra.Tool` pipeline and group them in a
`Terra.ToolRegistry`:

```elixir
defmodule MyTools do
  use Terra.ToolRegistry
  import Terra.Tool

  @impl true
  def tools do
    [
      tool("get_weather")
      |> desc("Get current weather for a city")
      |> param(:city, :string, required: true, desc: "City name")
      |> aging(expiry: 4)
      |> expiry_message(~S"<%= @input[:city] %> weather data expired — re-fetch if needed.")
    ]
  end

  @impl true
  def execute("get_weather", %{city: city}, state) do
    {:ok, %{temp: 72, city: city, conditions: "sunny"}, state}
  end
end
```

Register it in your agent's `init/1` via `registries: [MyTools]`. Terra executes
tool calls eagerly during streaming; handle the tool-use loop in
`handle_response/2` by appending results to history and re-invoking with the
`[:invoke]` action. Tool calls with no arguments (empty or whitespace input) are
treated as `%{}`, so required-field and default handling still apply.

### Documents

`Terra.Document` represents large, stable content that benefits from prompt
caching:

```elixir
doc = Terra.Document.new("forecast", "User's regional forecast", text, cache: :ephemeral)

state = Terra.Document.put(state, doc)

# In context/2:
Terra.Context.document(ctx, Terra.Document.get(state, "forecast"))
```

### Context Aging

Tool results are aged automatically based on distance from the most recent
assistant turn:

```elixir
tool("get_weather")
|> aging(expiry: 4, pruning: 8)
|> expiry_message(~S"Weather for <%= @input[:city] %> expired — re-fetch if needed.")
|> result_template(~S"Weather in <%= @input[:city] %>: <%= inspect(@result) %>")
```

- **Active** (distance < `expiry`) — full `result_template` shown to the LLM.
- **Expired** (`expiry` ≤ distance < `pruning`) — `expiry_message` shown instead.
- **Pruned** (distance ≥ `pruning`) — removed from context entirely.

### Multi-Agent Sessions

Use `Terra.Session` to run multiple agents that share a `Terra.Kernel`:

```elixir
{:ok, sup} = Terra.Session.start_link(
  buffers: [:plan, :analysis],
  agents: [
    {ConversationAgent, %{user_id: "123"}},
    {PlannerAgent, %{}},
    {AnalysisAgent, %{}}
  ]
)

[conv, planner, analysis] = Terra.Session.agents(sup)
kernel = Terra.Session.kernel(sup)
```

The supervisor uses a `:rest_for_one` strategy — if the kernel crashes, all
agents restart with a fresh kernel.

### Kernel

The kernel is a shared document store with named buffer slots, locking, and
pub/sub. Agents read, write, and coordinate through it:

```elixir
# Read / write
doc = Terra.Kernel.read(kernel, :plan)
:ok = Terra.Kernel.write(kernel, :plan, doc)

# Lock → work → write_and_unlock (atomic). Attach opaque metadata on lock.
:ok = Terra.Kernel.lock(kernel, :plan, %{focus: "intro", request_id: "abc"})
:ok = Terra.Kernel.write_and_unlock(kernel, :plan, doc)

# Refine slot metadata mid-loop without touching the lock (shallow merge)
:ok = Terra.Kernel.set_metadata(kernel, :plan, %{step: 2})

# Lightweight status: lock state, updated_at, content size, and metadata
{:ok, %{state: :idle, metadata: %{focus: "intro", step: 2}}} = Terra.Kernel.status(kernel, :plan)

# Subscribe to changes — receives {:kernel_update, :plan, doc}
Terra.Kernel.subscribe(kernel)
```

Locks are automatically released if the holding process exits. Metadata is opaque
to the kernel — callers attach whatever context they want (working focus, intent,
request id) and read it back via `status/2`. It persists across lock release.

## Providers

Terra ships three streaming providers, selected per-agent in `init/1`:

```elixir
provider: {Terra.Provider.Anthropic, %{}}  # Claude — Messages API
provider: {Terra.Provider.OpenAI, %{}}     # GPT / o-series — Chat Completions
provider: {Terra.Provider.Google, %{}}     # Gemini — Generative AI API
```

Provider config is passed through the `model` map returned by `context/2`:

```elixir
|> Terra.Context.model(%{
  model: "claude-sonnet-4-5-20250929",
  max_tokens: 4096,
  config: %{
    api_key: System.get_env("ANTHROPIC_API_KEY"),
    base_url: "https://api.anthropic.com",  # optional
    receive_timeout: 120_000,               # optional
    finch_name: MyApp.Finch                 # optional — use a named Finch pool
  }
})
```

Pass `finch_name:` to route requests through your application's own Finch pool for
connection reuse and tuning; omit it to use Req's default pool.

## Testing

```bash
mix deps.get
mix test            # unit tests
mix test --include live   # also run live provider tests (requires API keys)
```

Live provider tests are tagged `:live` and excluded by default. They read
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `GOOGLE_API_KEY` from the environment.

## Documentation

- [`guides/getting-started.md`](guides/getting-started.md) — full walkthrough.
- [`guides/cheatsheet.cheatmd`](guides/cheatsheet.cheatmd) — quick API reference.

Generate the full API docs with `mix docs`.

## License

Terra is released under the [Apache License 2.0](LICENSE).

Copyright © 2026 Anuvaya Labs.
