# Getting Started

Terra is an LLM agent framework built on Elixir's `:gen_statem`. This guide
walks you through building your first agent, adding tools, using documents,
and coordinating multiple agents.

## Installation

Add Terra to your `mix.exs` dependencies:

```elixir
defp deps do
  [
    {:terra, github: "anuvaya/terra"}
  ]
end
```

## Minimal Agent

A Terra agent needs four callbacks:

- `init/1` — set up initial state, provider, and registries
- `context/2` — build the LLM context window for each invocation
- `handle_input/3` — handle external messages
- `handle_response/2` — handle LLM responses
- `handle_stream_event/3` — handle individual stream events

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
    assistant = %{role: "assistant", content: text}
    history = state.data.history ++ [assistant]
    {:next_state, :idle, %{state | data: %{state.data | history: history}}}
  end
end
```

Start and interact with the agent:

```elixir
{:ok, pid} = MyAgent.start_link(%{})
Terra.Agent.send_input(pid, {:message, "Hello!"})
```

## Adding Tools

Define tools using a `Terra.ToolRegistry`:

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

Register it in your agent's `init/1`:

```elixir
def init(_args) do
  {:ok, :idle, %{history: []},
   provider: {Terra.Provider.Anthropic, %{}},
   registries: [MyTools]}
end
```

When the LLM calls a tool, Terra executes it eagerly during streaming. Your
`handle_response/2` receives the results in `resp.tool_results`. Handle the
tool-use loop by appending results to history and re-invoking:

```elixir
def handle_response(%{stop_reason: "tool_use"} = resp, state) do
  assistant = %{role: "assistant", content: resp.content_blocks}

  tool_results =
    Enum.map(resp.tool_results, fn tr ->
      content = case tr.result do
        {:ok, val} -> val
        {:error, reason} -> "Error: #{inspect(reason)}"
      end
      %{type: "tool_result", tool_use_id: tr.tool_use_id, content: content}
    end)

  history = state.data.history ++ [assistant, %{role: "user", content: tool_results}]
  state = %{state | data: %{state.data | history: history}}
  {:next_state, :thinking, state, [:invoke]}
end

def handle_response(resp, state) do
  text = Enum.find_value(resp.content_blocks, "", fn
    %{type: "text", text: t} -> t
    _ -> nil
  end)
  assistant = %{role: "assistant", content: text}
  history = state.data.history ++ [assistant]
  {:next_state, :idle, %{state | data: %{state.data | history: history}}}
end
```

## Using Documents

`Terra.Document` represents large, stable content that benefits from prompt
caching. Store documents on the agent state and inject them into context:

```elixir
def handle_input(:idle, {:set_forecast, forecast_text}, state) do
  doc = Terra.Document.new("forecast", "User's regional weather forecast", forecast_text, cache: :ephemeral)
  {:keep_state, Terra.Document.put(state, doc)}
end

def context(_state_name, state) do
  Terra.Context.new()
  |> Terra.Context.system("You are a weather forecaster.")
  |> Terra.Context.document(Terra.Document.get(state, "forecast"))
  |> Terra.Context.history(state.data.history)
  |> Terra.Context.model(%{model: "claude-sonnet-4-5-20250929", max_tokens: 4096})
  |> Terra.Context.age_tools(state)
  |> Terra.Context.build()
end
```

## Multi-Agent Sessions

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

# Get agent pids
[conv, planner, analysis] = Terra.Session.agents(sup)

# Agents read/write kernel slots
kernel = Terra.Session.kernel(sup)
Terra.Kernel.write(kernel, :plan, some_document)
doc = Terra.Kernel.read(kernel, :plan)
```

Agents can subscribe to kernel updates:

```elixir
def init(args) do
  if args[:kernel], do: Terra.Kernel.subscribe(args.kernel)
  {:ok, :idle, %{history: []},
   provider: {Terra.Provider.Anthropic, %{}},
   kernel: args[:kernel],
   registries: []}
end

def handle_info(_state, {:kernel_update, :plan, doc}, state) do
  # React to changes from other agents
  {:keep_state, Terra.Document.put(state, doc)}
end
```

## Context Aging

Tool results are automatically aged based on distance from the most recent
assistant turn. Configure aging per-tool:

```elixir
tool("get_weather")
|> desc("Get current weather")
|> param(:city, :string, required: true, desc: "City name")
|> aging(expiry: 4, pruning: 8)
|> expiry_message(~S"Weather for <%= @input[:city] %> expired — re-fetch if needed.")
|> result_template(~S"Weather in <%= @input[:city] %>: <%= inspect(@result) %>")
```

- **Active** (distance < `expiry`) — full `result_template` shown to LLM
- **Expired** (`expiry` <= distance < `pruning`) — `expiry_message` shown instead
- **Pruned** (distance >= `pruning`) — removed from context entirely
