defmodule Terra.Agent do
  @moduledoc """
  Behaviour and runtime for LLM agents built on `:gen_statem`.

  `use Terra.Agent` generates `start_link/1,2`, `child_spec/1`, and default
  implementations of optional callbacks. The consumer implements callbacks
  to control state transitions, streaming behaviour, and response handling.

  See `Terra.Agent.State` for the state struct passed to all callbacks,
  `Terra.Context` for context building with aging, and `Terra.ToolRegistry`
  for tool definitions.

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
          assistant = %{role: "assistant", content: resp.content_blocks}

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

  ## Lifecycle

  1. Consumer calls `MyAgent.start_link(args)` — gen_statem boots, calls `init/1`
  2. External code calls `Terra.Agent.send_input(pid, msg)` — triggers `handle_input/3`
  3. Consumer returns `[:invoke]` action — Terra calls `context/2`, then `provider.stream/2`
  4. Stream events arrive as `{:stream, ref, event}` — each forwarded to `handle_stream_event/3`
  5. On `:message_stop` — Terra builds a response map and calls `handle_response/2`
  6. Consumer decides: transition state, re-invoke, set timeouts, or stop

  ## Actions

  Consumer callbacks can return a list of actions:

  - `:invoke` — build context and start a new provider stream
  - `:cancel_stream` — cancel the active stream (clears `stream_ref`)
  - `{:state_timeout, ms, event}` — fire `handle_timeout/3` after `ms` milliseconds
  - `{:generic_timeout, name, ms, event}` — named timeout, fire `handle_timeout/3`

  ## Response Map

  The `handle_response/2` callback receives:

      %{
        stop_reason: "end_turn" | "tool_use" | "max_tokens" | ...,
        usage: %{input_tokens: integer, output_tokens: integer},
        content_blocks: [%{type: "text", text: "..."} | %{type: "tool_use", ...} | ...],
        tool_results: [%{tool_use_id: "...", name: "...", result: {:ok, term} | {:error, term}, is_error: boolean()}]
      }

  Completed tool blocks are eagerly executed on `content_block_stop`. If a
  stream reaches `:message_stop` before a tool block closes, Terra preserves
  the assistant `tool_use` block and synthesizes an error `tool_result` so the
  consumer can recover without executing partial input.

  ## Error Handling

  When a provider error occurs (either at invoke time or mid-stream),
  Terra calls `c:handle_error/3` instead of crashing the process:

  - `{:invoke_error, reason}` — `provider.stream/2` returned `{:error, reason}`
  - `{:stream_error, reason}` — `{:error, reason}` event arrived during streaming,
    or a linked process (e.g. the streaming Task) crashed mid-stream
  - `{:linked_crash, pid, reason}` — a linked process crashed outside of streaming

  Stream state (`stream_ref`, `_stream_meta`) is cleaned up before the callback
  is invoked, so the consumer can safely return `[:invoke]` to retry.

  Terra intercepts `{:EXIT, ...}` messages at the framework level (since it
  enables `trap_exit`). Normal exits are silently ignored. Crash exits are
  routed to `handle_error/3` — if mid-stream, wrapped as `{:stream_error, ...}`.

  Default: `{:stop, {:error, error}, state}` — matches pre-callback behaviour.

  ## Optional Callbacks

  - `handle_error/3` — provider errors (default: stop the process)
  - `handle_info/3` — arbitrary process messages (default: ignore)
  - `handle_timeout/3` — timeout events (default: ignore)
  """

  @type state_name :: atom()
  @type actions :: [action()]
  @type action ::
          :invoke
          | :cancel_stream
          | {:state_timeout, non_neg_integer(), term()}
          | {:generic_timeout, atom(), non_neg_integer(), term()}

  @callback init(args :: map()) ::
              {:ok, state_name(), data :: map(), keyword()}
              | {:stop, term()}

  @callback context(state_name(), State.t()) :: {Terra.Context.t(), State.t()}

  @callback handle_input(state_name(), input :: term(), State.t()) ::
              {:next_state, state_name(), State.t()}
              | {:next_state, state_name(), State.t(), actions()}
              | {:keep_state, State.t()}
              | {:keep_state, State.t(), actions()}
              | {:stop, term(), State.t()}

  @callback handle_stream_event(state_name(), Terra.Provider.event(), State.t()) ::
              {:keep_state, State.t()}
              | {:keep_state, State.t(), actions()}
              | {:next_state, state_name(), State.t()}
              | {:next_state, state_name(), State.t(), actions()}
              | {:stop, term(), State.t()}

  @callback handle_response(response :: map(), State.t()) ::
              {:next_state, state_name(), State.t()}
              | {:next_state, state_name(), State.t(), actions()}
              | {:keep_state, State.t()}
              | {:keep_state, State.t(), actions()}
              | {:stop, term(), State.t()}

  @callback handle_error(state_name(), error :: term(), State.t()) ::
              {:keep_state, State.t()}
              | {:keep_state, State.t(), actions()}
              | {:next_state, state_name(), State.t()}
              | {:next_state, state_name(), State.t(), actions()}
              | {:stop, term(), State.t()}

  @callback handle_info(state_name(), msg :: term(), State.t()) ::
              {:keep_state, State.t()}
              | {:keep_state, State.t(), actions()}
              | {:next_state, state_name(), State.t()}
              | {:next_state, state_name(), State.t(), actions()}
              | {:stop, term(), State.t()}

  @callback handle_timeout(state_name(), event :: term(), State.t()) ::
              {:keep_state, State.t()}
              | {:keep_state, State.t(), actions()}
              | {:next_state, state_name(), State.t()}
              | {:next_state, state_name(), State.t(), actions()}
              | {:stop, term(), State.t()}

  @callback terminate(reason :: term(), State.t()) :: term()

  @optional_callbacks [
    handle_stream_event: 3,
    handle_error: 3,
    handle_info: 3,
    handle_timeout: 3,
    terminate: 2
  ]

  # -- Public API --

  @doc """
  Send an input message to a running agent.

  The input is delivered as an asynchronous cast and dispatched to the
  agent's `c:handle_input/3` callback. Any term can be sent — the
  consumer defines the input protocol.

  ## Example

      Terra.Agent.send_input(pid, {:message, "What is the weather like?"})
  """
  @spec send_input(pid() | GenServer.name(), term()) :: :ok
  def send_input(pid, input) do
    :gen_statem.cast(pid, {:input, input})
  end

  # -- __using__ macro --

  defmacro __using__(_opts) do
    quote do
      @behaviour Terra.Agent

      def start_link(args, opts \\ []) do
        case Keyword.pop(opts, :name) do
          {nil, opts} ->
            :gen_statem.start_link(Terra.Agent.Server, {__MODULE__, args}, opts)

          {name, opts} ->
            :gen_statem.start_link(name, Terra.Agent.Server, {__MODULE__, args}, opts)
        end
      end

      def child_spec(args) do
        %{
          id: __MODULE__,
          start: {__MODULE__, :start_link, [args]},
          type: :worker,
          restart: :permanent
        }
      end

      def handle_stream_event(_state_name, _event, state), do: {:keep_state, state}
      def handle_error(_state_name, error, state), do: {:stop, {:error, error}, state}
      def handle_info(_state_name, _msg, state), do: {:keep_state, state}
      def handle_timeout(_state_name, _event, state), do: {:keep_state, state}
      def terminate(_reason, _state), do: :ok

      defoverridable handle_stream_event: 3,
                     handle_error: 3,
                     handle_info: 3,
                     handle_timeout: 3,
                     terminate: 2
    end
  end

  # -- State struct --

  defmodule State do
    @moduledoc """
    State struct carried through all `Terra.Agent` callbacks.

    The consumer reads and updates this struct to drive agent behaviour.
    Custom data lives in the `data` field; all other fields are managed
    by Terra internally.

    ## Fields

    - `data` — consumer's custom data map (whatever you return from `c:Terra.Agent.init/1`)
    - `handler` — the callback module (set by Terra, do not modify)
    - `provider` — `{module, config}` tuple identifying the LLM provider
    - `registries` — list of `Terra.ToolRegistry` modules for tool execution
    - `kernel` — pid of the shared `Terra.Kernel` (`nil` outside a `Terra.Session`)
    - `documents` — `%{title => Terra.Document.t()}` map managed via
      `Terra.Document.put/2`, `Terra.Document.get/2`, `Terra.Document.delete/2`
    - `messages` — internal message buffer (prefer `data` for history management)
    - `stream_ref` — reference of the active provider stream (`nil` when idle)
    - `turn_count` — number of completed LLM turns (increments on each `c:Terra.Agent.handle_response/2`)
    - `tokens` — `%{input: n, output: n}` cumulative token usage across all turns
    """

    @type t :: %__MODULE__{
            data: map(),
            handler: module(),
            provider: {module(), term()},
            registries: [module()],
            kernel: pid() | nil,
            documents: %{String.t() => Terra.Document.t()},
            messages: [map()],
            stream_ref: reference() | nil,
            turn_count: non_neg_integer(),
            tokens: %{input: non_neg_integer(), output: non_neg_integer()},
            _stream_meta: map(),
            _req_id: String.t() | nil
          }

    defstruct data: %{},
              handler: nil,
              provider: nil,
              registries: [],
              kernel: nil,
              documents: %{},
              messages: [],
              stream_ref: nil,
              turn_count: 0,
              tokens: %{input: 0, output: 0},
              _stream_meta: %{},
              _req_id: nil
  end

  # -- Server (gen_statem) --

  defmodule Server do
    @moduledoc false

    alias Terra.Agent.State
    require Logger

    @behaviour :gen_statem

    @impl true
    def callback_mode, do: :handle_event_function

    @impl true
    def init({handler, args}) do
      Process.flag(:trap_exit, true)

      case handler.init(args) do
        {:ok, state_name, data, opts} ->
          {provider_mod, provider_config} = Keyword.fetch!(opts, :provider)

          kernel = Map.get(args, :kernel)
          if kernel, do: Terra.Kernel.subscribe(kernel)

          state = %Terra.Agent.State{
            data: data,
            handler: handler,
            provider: {provider_mod, provider_config},
            registries: Keyword.get(opts, :registries, []),
            kernel: kernel,
            messages: Keyword.get(opts, :messages, [])
          }

          Terra.Telemetry.agent_init(handler, state_name, data)

          case Keyword.get(opts, :actions, []) do
            [] -> {:ok, state_name, state}
            actions -> {:ok, state_name, state, translate_actions(actions)}
          end

        {:stop, reason} ->
          {:stop, reason}
      end
    end

    @impl true
    # -- Input cast --
    def handle_event(:cast, {:input, input}, state_name, %State{} = state) do
      req_id = generate_req_id()
      state = %{state | _req_id: req_id}

      Terra.Telemetry.agent_input(state.handler, state_name, input, req_id)

      state.handler
      |> apply(:handle_input, [state_name, input, state])
      |> translate_and_emit(state_name, state)
    end

    # -- Internal: invoke --
    def handle_event(:internal, :invoke, state_name, %State{} = state) do
      %{handler: handler, provider: {provider_mod, _config}} = state

      req_id = generate_req_id()
      state = %{state | _req_id: req_id}

      {context, state} = handler.context(state_name, state)

      tools = collect_tools(state)

      params =
        context.model
        |> Map.put(:system, context.system)
        |> Map.put(:messages, context.messages)
        |> Map.put(:tools, tools)

      Terra.Telemetry.invoke_start(handler, state_name, %{
        messages: params.messages,
        tool_count: length(tools),
        tool_names: Enum.map(tools, & &1[:name]),
        turn_count: state.turn_count,
        has_system: params.system != nil,
        max_tokens: Map.get(params, :max_tokens)
      }, req_id)

      case provider_mod.stream(self(), params) do
        {:ok, ref} ->
          meta = Map.put(state._stream_meta, :invoke_start_time, System.monotonic_time())
          {:keep_state, %{state | stream_ref: ref, _stream_meta: meta}}

        {:error, reason} ->
          Terra.Telemetry.invoke_error(handler, :invoke_error, reason, req_id)

          state.handler
          |> apply(:handle_error, [state_name, {:invoke_error, reason}, state])
          |> translate_and_emit(state_name, state)
      end
    end

    # -- Internal: cancel_stream --
    def handle_event(:internal, :cancel_stream, state_name, %State{} = state) do
      case state.stream_ref do
        nil ->
          :keep_state_and_data

        ref ->
          Terra.Telemetry.cancel_stream(state.handler, state_name, state._req_id)
          {provider_mod, _config} = state.provider
          provider_mod.cancel(ref)
          {:keep_state, %{state | stream_ref: nil, _stream_meta: %{}}}
      end
    end

    # -- Stream events (matching ref) --
    def handle_event(:info, {:stream, ref, event}, state_name, %State{stream_ref: ref} = state) do
      case event do
        # message_start carries the initial message with input token usage
        {:message_start, %{usage: usage} = _message} ->
          input_tokens = Map.get(usage, :input_tokens, 0)
          meta = Map.put(state._stream_meta, :input_tokens, input_tokens)
          state = %{state | _stream_meta: meta}

          state.handler
          |> apply(:handle_stream_event, [state_name, event, state])
          |> translate_and_emit(state_name, state)

        # message_delta carries stop_reason and output token usage
        {:message_delta, delta, usage} ->
          output_tokens = Map.get(usage, :output_tokens, 0)
          stop_reason = Map.get(delta, :stop_reason)

          meta =
            state._stream_meta
            |> Map.put(:output_tokens, output_tokens)
            |> Map.put(:stop_reason, stop_reason)

          state = %{state | _stream_meta: meta}

          state.handler
          |> apply(:handle_stream_event, [state_name, event, state])
          |> translate_and_emit(state_name, state)

        # message_stop finalizes the turn — always calls handle_response
        :message_stop ->
          meta = state._stream_meta
          input_tokens = Map.get(meta, :input_tokens, 0)
          output_tokens = Map.get(meta, :output_tokens, 0)
          stop_reason = Map.get(meta, :stop_reason)
          invoke_start_time = Map.get(meta, :invoke_start_time)

          {content_blocks, tool_results} = finalize_response_content(meta, stop_reason)

          tool_names = tool_results |> Enum.map(& &1.name)
          block_types = content_blocks |> Enum.map(& &1.type) |> Enum.frequencies()

          usage = %{input_tokens: input_tokens, output_tokens: output_tokens}
          duration = if invoke_start_time, do: System.monotonic_time() - invoke_start_time, else: 0
          Terra.Telemetry.invoke_stop(state.handler, state_name, stop_reason, usage, duration, %{
            tool_names: tool_names,
            block_types: block_types,
            tool_count: length(tool_results),
            turn_count: state.turn_count
          }, state._req_id)

          state = %{
            state
            | stream_ref: nil,
              turn_count: state.turn_count + 1,
              tokens: %{
                input: state.tokens.input + input_tokens,
                output: state.tokens.output + output_tokens
              },
              _stream_meta: %{}
          }

          response = %{
            stop_reason: stop_reason,
            usage: usage,
            content_blocks: content_blocks,
            tool_results: tool_results
          }

          state.handler
          |> apply(:handle_response, [response, state])
          |> translate_and_emit(state_name, state)

        {:error, reason} ->
          Terra.Telemetry.invoke_error(state.handler, :stream_error, reason, state._req_id)
          state = %{state | stream_ref: nil, _stream_meta: %{}}

          state.handler
          |> apply(:handle_error, [state_name, {:stream_error, reason}, state])
          |> translate_and_emit(state_name, state)

        # content_block_start — begin accumulating a new block
        {:content_block_start, index, %{type: "tool_use"} = block} ->
          blocks = Map.get(state._stream_meta, :content_blocks, %{})

          acc = %{
            type: "tool_use",
            id: block.id,
            name: block.name,
            input_json: "",
            closed?: false
          }

          meta = Map.put(state._stream_meta, :content_blocks, Map.put(blocks, index, acc))
          state = %{state | _stream_meta: meta}

          state.handler
          |> apply(:handle_stream_event, [state_name, event, state])
          |> translate_and_emit(state_name, state)

        {:content_block_start, index, %{type: type}} ->
          blocks = Map.get(state._stream_meta, :content_blocks, %{})

          acc =
            case type do
              "text" -> %{type: "text", text: ""}
              "thinking" -> %{type: "thinking", thinking: "", signature: ""}
              _ -> %{type: type}
            end

          meta = Map.put(state._stream_meta, :content_blocks, Map.put(blocks, index, acc))
          state = %{state | _stream_meta: meta}

          state.handler
          |> apply(:handle_stream_event, [state_name, event, state])
          |> translate_and_emit(state_name, state)

        # content_block_delta — accumulate into the block
        {:content_block_delta, index, %{type: "input_json_delta", partial_json: json}} ->
          state =
            update_block(state, index, fn block ->
              %{block | input_json: block.input_json <> json}
            end)

          state.handler
          |> apply(:handle_stream_event, [state_name, event, state])
          |> translate_and_emit(state_name, state)

        {:content_block_delta, index, %{type: "text_delta", text: text}} ->
          state =
            update_block(state, index, fn block ->
              %{block | text: block.text <> text}
            end)

          state.handler
          |> apply(:handle_stream_event, [state_name, event, state])
          |> translate_and_emit(state_name, state)

        {:content_block_delta, index, %{type: "thinking_delta", thinking: text}} ->
          state =
            update_block(state, index, fn block ->
              %{block | thinking: block.thinking <> text}
            end)

          state.handler
          |> apply(:handle_stream_event, [state_name, event, state])
          |> translate_and_emit(state_name, state)

        {:content_block_delta, index, %{type: "signature_delta", signature: sig}} ->
          state =
            update_block(state, index, fn block ->
              %{block | signature: block.signature <> sig}
            end)

          state.handler
          |> apply(:handle_stream_event, [state_name, event, state])
          |> translate_and_emit(state_name, state)

        # content_block_stop — eagerly execute tool_use blocks
        {:content_block_stop, index} ->
          state =
            update_block(state, index, fn
              %{type: "tool_use"} = block -> %{block | closed?: true}
              block -> block
            end)

          blocks = Map.get(state._stream_meta, :content_blocks, %{})

          state =
            case Map.get(blocks, index) do
              %{type: "tool_use"} = block ->
                reg = Enum.find(state.registries, & &1.__has_tool__?(block.name, state))

                {result, updated_state} =
                  case reg do
                    nil ->
                      {{:error, "unknown tool: #{block.name}"}, state}

                    reg ->
                      Terra.Telemetry.tool_start(block.name, reg, state._req_id)
                      tool_start_time = System.monotonic_time()

                      {res, new_state} =
                        try do
                          case reg.handle_execution(block.name, block.input_json, state) do
                            {:ok, value, new_state} -> {{:ok, value}, new_state}
                            {:error, reason, new_state} -> {{:error, reason}, new_state}
                          end
                        rescue
                          e ->
                            Logger.error("Tool execution crashed: #{block.name} — #{Exception.message(e)}")
                            {{:error, "Tool execution failed: #{Exception.message(e)}"}, state}
                        end

                      tool_duration = System.monotonic_time() - tool_start_time
                      result_type = if match?({:ok, _}, res), do: :ok, else: :error
                      Terra.Telemetry.tool_stop(block.name, reg, result_type, tool_duration, state._req_id)

                      {res, new_state}
                  end

                tool_result = %{
                  tool_use_id: block.id,
                  name: block.name,
                  result: result,
                  is_error: match?({:error, _}, result)
                }

                tool_results =
                  Map.get(updated_state._stream_meta, :tool_results, []) ++ [tool_result]

                %{
                  updated_state
                  | _stream_meta: Map.put(updated_state._stream_meta, :tool_results, tool_results)
                }

              _ ->
                state
            end

          state.handler
          |> apply(:handle_stream_event, [state_name, event, state])
          |> translate_and_emit(state_name, state)

        # All other events (ping, etc.) → consumer
        _other ->
          state.handler
          |> apply(:handle_stream_event, [state_name, event, state])
          |> translate_and_emit(state_name, state)
      end
    end

    # -- Stale stream messages (wrong ref) --
    def handle_event(:info, {:stream, _stale_ref, _event}, _state_name, _state) do
      :keep_state_and_data
    end

    # -- State timeout --
    def handle_event(:state_timeout, event, state_name, %State{} = state) do
      state.handler
      |> apply(:handle_timeout, [state_name, event, state])
      |> translate_and_emit(state_name, state)
    end

    # -- Generic (named) timeout --
    def handle_event({:timeout, _name}, event, state_name, %State{} = state) do
      state.handler
      |> apply(:handle_timeout, [state_name, event, state])
      |> translate_and_emit(state_name, state)
    end

    # -- Linked process exits (trap_exit is on) --
    def handle_event(:info, {:EXIT, _pid, :normal}, _state_name, _state) do
      :keep_state_and_data
    end

    def handle_event(:info, {:EXIT, pid, reason}, state_name, %State{} = state) do
      if state.stream_ref do
        # Mid-stream crash (likely the provider Task) — treat as stream error
        state = %{state | stream_ref: nil, _stream_meta: %{}}

        state.handler
        |> apply(:handle_error, [state_name, {:stream_error, {:linked_crash, pid, reason}}, state])
        |> translate_and_emit(state_name, state)
      else
        # Not mid-stream — inform consumer via handle_error
        state.handler
        |> apply(:handle_error, [state_name, {:linked_crash, pid, reason}, state])
        |> translate_and_emit(state_name, state)
      end
    end

    # -- Arbitrary info messages --
    def handle_event(:info, msg, state_name, %State{} = state) do
      state.handler
      |> apply(:handle_info, [state_name, msg, state])
      |> translate_and_emit(state_name, state)
    end

    def handle_event(_type, _content, _state_name, _state) do
      :keep_state_and_data
    end

    @impl true
    def terminate(reason, _state_name, %State{} = state) do
      state.handler.terminate(reason, state)
    end

    defp collect_tools(%State{registries: registries} = state) do
      registries
      |> Enum.flat_map(& &1.tools(state))
      |> Enum.map(&Terra.Tool.build/1)
    end

    defp update_block(%State{} = state, index, fun) do
      blocks = Map.get(state._stream_meta, :content_blocks, %{})

      case Map.get(blocks, index) do
        nil ->
          state

        block ->
          blocks = Map.put(blocks, index, fun.(block))
          %{state | _stream_meta: Map.put(state._stream_meta, :content_blocks, blocks)}
      end
    end

    defp finalize_response_content(meta, stop_reason) do
      blocks =
        meta
        |> Map.get(:content_blocks, %{})
        |> Enum.sort_by(fn {idx, _} -> idx end)

      actual_tool_results = Map.get(meta, :tool_results, [])

      tool_result_ids =
        actual_tool_results
        |> Enum.map(& &1.tool_use_id)
        |> MapSet.new()

      {content_blocks, synthetic_tool_results} =
        Enum.reduce(blocks, {[], []}, fn {_idx, block}, {content_acc, tool_acc} ->
          case block do
            %{type: "tool_use", id: id} = tool_block ->
              content_block = finalize_block(tool_block)

              if MapSet.member?(tool_result_ids, id) do
                {[content_block | content_acc], tool_acc}
              else
                log_incomplete_tool_use(tool_block, stop_reason)
                synthetic_tool_result = build_incomplete_tool_result(tool_block, stop_reason)
                {[content_block | content_acc], [synthetic_tool_result | tool_acc]}
              end

            other ->
              {[finalize_block(other) | content_acc], tool_acc}
          end
        end)

      {Enum.reverse(content_blocks), actual_tool_results ++ Enum.reverse(synthetic_tool_results)}
    end

    defp log_incomplete_tool_use(block, stop_reason) do
      partial_json = Map.get(block, :input_json, "")

      Logger.warning(
        "Incomplete tool_use block at message_stop: #{block.name} (#{block.id}) stop=#{stop_reason || "unknown"}",
        tool_use_id: block.id,
        tool_name: block.name,
        stop_reason: stop_reason,
        tool_block_closed: Map.get(block, :closed?, false),
        partial_json_size: byte_size(partial_json),
        partial_json_preview: String.slice(partial_json, 0, 200)
      )
    end

    defp build_incomplete_tool_result(block, stop_reason) do
      reason =
        if Map.get(block, :closed?, false) do
          "Tool call failed internally: tool block closed but no tool result was recorded before message_stop."
        else
          "Tool call incomplete: stream ended with stop_reason=#{stop_reason || "unknown"} before content_block_stop. Re-emit the tool call with complete JSON input."
        end

      %{
        tool_use_id: block.id,
        name: block.name,
        result: {:error, reason},
        is_error: true
      }
    end

    defp finalize_block(%{type: "tool_use"} = block) do
      input =
        case Jason.decode(block.input_json, keys: :atoms) do
          {:ok, parsed} -> parsed
          _ -> %{}
        end

      %{type: "tool_use", id: block.id, name: block.name, input: input}
    end

    defp finalize_block(%{type: "text"} = block) do
      %{type: "text", text: block.text}
    end

    defp finalize_block(%{type: "thinking"} = block) do
      %{type: "thinking", thinking: block.thinking, signature: block.signature}
    end

    defp finalize_block(block), do: block

    defp translate_return({:next_state, new_state_name, new_state}) do
      {:next_state, new_state_name, new_state}
    end

    defp translate_return({:next_state, new_state_name, new_state, actions}) do
      {:next_state, new_state_name, new_state, translate_actions(actions)}
    end

    defp translate_return({:keep_state, new_state}) do
      {:keep_state, new_state}
    end

    defp translate_return({:keep_state, new_state, actions}) do
      {:keep_state, new_state, translate_actions(actions)}
    end

    defp translate_return({:stop, reason, new_state}) do
      {:stop, reason, new_state}
    end

    defp translate_and_emit(result, current_state_name, state) do
      translated = translate_return(result)
      Terra.Telemetry.maybe_state_change(current_state_name, translated, state)
      translated
    end

    defp generate_req_id do
      <<a::32, b::16, _::4, c::12, _::2, d::62>> = :crypto.strong_rand_bytes(16)
      <<a::32, b::16, 4::4, c::12, 2::2, d::62>>
      |> Base.encode16(case: :lower)
      |> then(fn hex ->
        <<a::binary-size(8), b::binary-size(4), c::binary-size(4), d::binary-size(4), e::binary>> = hex
        "#{a}-#{b}-#{c}-#{d}-#{e}"
      end)
    end

    defp translate_actions(actions) when is_list(actions) do
      Enum.map(actions, &translate_action/1)
    end

    defp translate_action(:invoke), do: {:next_event, :internal, :invoke}
    defp translate_action(:cancel_stream), do: {:next_event, :internal, :cancel_stream}

    defp translate_action({:state_timeout, ms, event}),
      do: {:state_timeout, ms, event}

    defp translate_action({:generic_timeout, name, ms, event}),
      do: {{:timeout, name}, ms, event}
  end
end
