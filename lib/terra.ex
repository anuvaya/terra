defmodule Terra do
  @moduledoc """
  Terra is an LLM agent framework built on Elixir's `:gen_statem`.

  It provides a behaviour-based approach to building stateful, streaming LLM agents
  with tool execution, context aging, and multi-agent coordination.

  ## Quick Start

      defmodule MyAgent do
        use Terra.Agent

        def init(args) do
          {:ok, :idle, %{history: []},
           provider: {Terra.Provider.Anthropic, %{}},
           registries: [MyTools]}
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

        def handle_response(%{stop_reason: "tool_use"} = resp, state) do
          # Build the assistant message from content blocks
          assistant = %{role: "assistant", content: resp.content_blocks}

          # Build tool_result entries for the next user turn
          tool_results =
            Enum.map(resp.tool_results, fn tr ->
              content = case tr.result do
                {:ok, val} -> val
                {:error, reason} -> "Error: \#{inspect(reason)}"
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
      end

  ## Architecture

  - `Terra.Agent` — Behaviour and `:gen_statem` runtime for LLM agents
  - `Terra.Agent.State` — State struct carried through all agent callbacks
  - `Terra.Context` — Builder for assembling the LLM context window with aging
  - `Terra.Tool` — Pipeline builder for tool definitions (params, aging, templates)
  - `Terra.ToolRegistry` — Behaviour for grouping and executing tools
  - `Terra.Provider` — Behaviour for streaming LLM providers (Anthropic, OpenAI, Google)
  - `Terra.Document` — Structured content blocks for context injection
  - `Terra.Session` — Supervisor for multi-agent sessions with shared state
  - `Terra.Kernel` — Shared document store for inter-agent communication

  ## Providers

  Terra ships with three streaming providers:

  - `Terra.Provider.Anthropic` — Claude models via the Messages API
  - `Terra.Provider.OpenAI` — GPT/o-series models via Chat Completions
  - `Terra.Provider.Google` — Gemini models via the Generative AI API

  All providers translate their native streaming format into Terra's unified event
  protocol, so agents work identically regardless of the underlying LLM.

  ## Multi-Agent Sessions

      {:ok, sup} = Terra.Session.start_link(
        buffers: [:plan, :analysis],
        agents: [
          {ConversationAgent, %{user_id: "123"}},
          {PlannerAgent, %{}},
          {AnalysisAgent, %{}}
        ]
      )

  Agents in a session share a `Terra.Kernel` for document exchange.
  The supervisor uses `:rest_for_one` strategy — if the kernel crashes,
  all agents restart with a fresh kernel.
  """
end
