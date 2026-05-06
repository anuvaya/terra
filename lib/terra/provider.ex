defmodule Terra.Provider do
  @moduledoc """
  Behaviour for LLM streaming providers.

  Implementations receive a caller pid and a params map, then send
  streaming events as `{:stream, ref, event}` to the caller. Terra ships
  with three providers:

  - `Terra.Provider.Anthropic` — Claude models via the Messages API
  - `Terra.Provider.OpenAI` — GPT/o-series via Chat Completions
  - `Terra.Provider.Google` — Gemini via the Generative AI API

  All providers translate their native streaming format into Terra's unified
  event protocol, so agents work identically regardless of the underlying LLM.

  ## Implementing a Custom Provider

      defmodule MyProvider do
        @behaviour Terra.Provider

        @impl true
        def stream(caller, params) do
          ref = make_ref()

          Task.start_link(fn ->
            # ... make HTTP request, parse response ...
            send(caller, {:stream, ref, {:message_start, %{id: "...", content: [], usage: %{input_tokens: 0, output_tokens: 0}}}})
            send(caller, {:stream, ref, {:content_block_start, 0, %{type: "text", text: ""}}})
            send(caller, {:stream, ref, {:content_block_delta, 0, %{type: "text_delta", text: "Hello"}}})
            send(caller, {:stream, ref, {:content_block_stop, 0}})
            send(caller, {:stream, ref, {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 1}}})
            send(caller, {:stream, ref, :message_stop})
          end)

          {:ok, ref}
        end

        @impl true
        def cancel(_ref), do: :ok
      end

  ## Params

  The params map contains:
  - `:system` — system prompt (string)
  - `:messages` — conversation history (list of maps)
  - `:tools` — tool definitions from registries (list of maps)
  - `:model`, `:max_tokens`, `:config` — from the consumer's `context/2` callback
  - Provider-specific fields (`:temperature`, `:top_p`, `:thinking`, etc.)

  ## Event Protocol

  Events mirror the Anthropic Messages streaming SSE protocol:

      {:message_start, %{id: _, content: [], usage: %{input_tokens: _}}}
      {:content_block_start, index, %{type: "text" | "thinking" | "tool_use", ...}}
      {:content_block_delta, index, %{type: "text_delta", text: _}}
      {:content_block_delta, index, %{type: "thinking_delta", thinking: _}}
      {:content_block_delta, index, %{type: "input_json_delta", partial_json: _}}
      {:content_block_delta, index, %{type: "signature_delta", signature: _}}
      {:content_block_stop, index}
      {:message_delta, %{stop_reason: _}, %{output_tokens: _}}
      :message_stop
      :ping
      {:error, %Terra.APIError{}}
  """

  @type ref :: reference()
  @type message :: map()

  @type params :: %{
          required(:system) => String.t(),
          required(:messages) => [message()],
          required(:tools) => [Terra.ToolRegistry.tool()],
          optional(atom()) => term()
        }

  @type event ::
          {:message_start, map()}
          | {:content_block_start, non_neg_integer(), map()}
          | {:content_block_delta, non_neg_integer(), map()}
          | {:content_block_stop, non_neg_integer()}
          | {:message_delta, map(), map()}
          | :message_stop
          | :ping
          | {:error, map()}

  @callback stream(caller :: pid(), params :: params()) ::
              {:ok, ref()} | {:error, term()}

  @callback cancel(ref()) :: :ok
end
