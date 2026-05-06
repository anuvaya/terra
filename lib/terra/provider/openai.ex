defmodule Terra.Provider.OpenAI do
  @moduledoc """
  OpenAI Chat Completions streaming provider.

  Translates OpenAI's delta-based streaming format into Terra's structured
  event protocol. OpenAI uses `data: {json}\\n\\n` lines with no `event:` field,
  terminated by `data: [DONE]`.

  ## Config

      def context(_state_name, _state) do
        Terra.Context.new()
        |> Terra.Context.system("You are helpful.")
        |> Terra.Context.model(%{
          model: "gpt-4o",
          max_tokens: 4096,
          config: %{
            api_key: System.get_env("OPENAI_API_KEY"),
            base_url: "https://api.openai.com",   # optional, swap for compatible APIs
            receive_timeout: 120_000                # optional
          }
        })
        |> Terra.Context.build()
      end

  ## Supported Parameters

  - `:model` — model ID (required)
  - `:max_tokens`, `:max_completion_tokens` — output token limits
  - `:system` — system prompt (injected as first message with role "system")
  - `:messages` — conversation history
  - `:tools` — tool definitions (formatted as `{type: "function", function: {...}}`)
  - `:tool_choice`, `:parallel_tool_calls` — tool selection control
  - `:temperature`, `:top_p`, `:frequency_penalty`, `:presence_penalty`
  - `:stop_sequences`, `:seed`, `:reasoning_effort`

  ## Translation

  | OpenAI                    | Terra Event                          |
  |---------------------------|--------------------------------------|
  | `delta.content`           | `{:content_block_delta, 0, text_delta}` |
  | `delta.tool_calls[i]`     | `{:content_block_start/delta, i+1, ...}` |
  | `finish_reason: "stop"`   | `stop_reason: "end_turn"`            |
  | `finish_reason: "tool_calls"` | `stop_reason: "tool_use"`        |
  | `finish_reason: "length"` | `stop_reason: "max_tokens"`          |
  | Usage-only chunk          | `message_delta` + `message_stop`     |
  """

  @behaviour Terra.Provider

  alias Terra.APIError

  @default_base_url "https://api.openai.com"
  @default_timeout 120_000

  @impl true
  def stream(caller, params) do
    ref = make_ref()
    config = Map.get(params, :config, %{})

    task =
      Task.start_link(fn ->
        run_stream(caller, ref, config, params)
      end)

    case task do
      {:ok, _pid} -> {:ok, ref}
      {:error, reason} -> {:error, reason}
    end
  end

  @impl true
  def cancel(_ref), do: :ok

  # ── Request Building ───────────────────────────────────

  defp run_stream(caller, ref, config, params) do
    api_key = Map.fetch!(config, :api_key)
    base_url = Map.get(config, :base_url, @default_base_url)
    timeout = Map.get(config, :receive_timeout, @default_timeout)

    body = build_body(params)
    headers = build_headers(api_key)

    Terra.Telemetry.provider_request(caller, body, headers)

    req =
      Req.new(
        base_url: base_url,
        url: "/v1/chat/completions",
        method: :post,
        headers: headers,
        body: Jason.encode!(body),
        receive_timeout: timeout,
        into: stream_handler(caller, ref)
      )

    case Req.request(req) do
      {:ok, %{status: status}} when status in 200..299 ->
        flush_buffer(caller, ref)
        :ok

      {:ok, resp} ->
        raw = Process.get(:sse_error_body, "")
        Process.delete(:sse_error_body)
        error = APIError.from_raw_body(resp.status, raw, Map.to_list(resp.headers))
        send(caller, {:stream, ref, {:error, error}})

      {:error, %Mint.TransportError{} = error} ->
        send(caller, {:stream, ref, {:error, %APIError{type: :connection_error, message: Exception.message(error)}}})

      {:error, reason} ->
        send(caller, {:stream, ref, {:error, %APIError{type: :connection_error, message: inspect(reason)}}})
    end
  end

  defp build_headers(api_key) do
    %{
      "content-type" => "application/json",
      "authorization" => "Bearer #{api_key}"
    }
  end

  defp build_body(params) do
    body = %{
      model: Map.fetch!(params, :model),
      stream: true,
      stream_options: %{include_usage: true}
    }

    optional_fields = [
      {:system, nil},
      {:messages, nil},
      {:max_tokens, :max_tokens},
      {:max_completion_tokens, :max_completion_tokens},
      {:temperature, :temperature},
      {:top_p, :top_p},
      {:frequency_penalty, :frequency_penalty},
      {:presence_penalty, :presence_penalty},
      {:stop_sequences, :stop},
      {:seed, :seed},
      {:reasoning_effort, :reasoning_effort}
    ]

    body =
      Enum.reduce(optional_fields, body, fn
        {:system, nil}, acc ->
          # system goes into messages, not body
          acc

        {:messages, nil}, acc ->
          # handled separately below
          acc

        {:stop_sequences, :stop}, acc ->
          case Map.get(params, :stop_sequences) do
            nil -> acc
            [] -> acc
            seqs -> Map.put(acc, :stop, seqs)
          end

        {param_key, body_key}, acc ->
          case Map.get(params, param_key) do
            nil -> acc
            value -> Map.put(acc, body_key, value)
          end
      end)

    # Build messages with system prompt
    messages =
      case Map.get(params, :system) do
        nil -> []
        "" -> []
        system -> [%{role: "system", content: system}]
      end ++ render_messages(Map.get(params, :messages, []))

    body = Map.put(body, :messages, messages)

    # Tools
    case Map.get(params, :tools, []) do
      [] ->
        body

      tools ->
        formatted = Enum.map(tools, fn tool ->
          %{
            type: "function",
            function: %{
              name: tool.name,
              description: tool.description,
              parameters: tool.input_schema
            }
          }
        end)

        body
        |> Map.put(:tools, formatted)
        |> maybe_put(:tool_choice, Map.get(params, :tool_choice))
        |> maybe_put(:parallel_tool_calls, Map.get(params, :parallel_tool_calls))
    end
  end

  defp maybe_put(map, _key, nil), do: map
  defp maybe_put(map, key, value), do: Map.put(map, key, value)

  # ── Message Translation ────────────────────────────────
  # Terra agents emit Anthropic-flavored content blocks (tool_use, tool_result,
  # thinking, plus Terra.Document structs). OpenAI Chat Completions wants:
  #   - assistant tool calls in a `tool_calls` array (not in content blocks)
  #   - tool results as separate `role: "tool"` messages with `tool_call_id`
  #   - documents flattened to text
  @doc false
  def render_messages(messages) do
    Enum.flat_map(messages, fn msg ->
      role = Map.get(msg, :role) || Map.get(msg, "role")
      content = Map.get(msg, :content) || Map.get(msg, "content")

      case {role, content} do
        {"assistant", content} when is_list(content) ->
          render_assistant_message(content)

        {"user", content} when is_list(content) ->
          render_user_message(content)

        _ ->
          [msg]
      end
    end)
  end

  defp render_assistant_message(content) do
    text =
      content
      |> Enum.filter(&match?(%{type: "text"}, &1))
      |> Enum.map(&Map.get(&1, :text, ""))
      |> Enum.join("")

    tool_calls =
      content
      |> Enum.filter(&match?(%{type: "tool_use"}, &1))
      |> Enum.map(fn block ->
        %{
          id: Map.get(block, :id),
          type: "function",
          function: %{
            name: Map.get(block, :name),
            arguments: Jason.encode!(Map.get(block, :input) || %{})
          }
        }
      end)

    base = %{role: "assistant"}

    base =
      case {text, tool_calls} do
        {"", []} -> Map.put(base, :content, "")
        {"", _} -> Map.put(base, :content, nil)
        {t, _} -> Map.put(base, :content, t)
      end

    case tool_calls do
      [] -> [base]
      _ -> [Map.put(base, :tool_calls, tool_calls)]
    end
  end

  defp render_user_message(content) do
    {tool_results, other} =
      Enum.split_with(content, &match?(%{type: "tool_result"}, &1))

    tool_msgs =
      Enum.map(tool_results, fn block ->
        %{
          role: "tool",
          tool_call_id: Map.get(block, :tool_use_id),
          content: stringify_tool_content(Map.get(block, :content))
        }
      end)

    user_text = render_user_content(other)

    user_msgs =
      case user_text do
        "" -> []
        t -> [%{role: "user", content: t}]
      end

    # OpenAI requires tool messages to immediately follow the assistant
    # turn that issued the tool calls — emit them before any new user text.
    tool_msgs ++ user_msgs
  end

  defp render_user_content(blocks) do
    blocks
    |> Enum.map(&render_user_block/1)
    |> Enum.reject(&(&1 == ""))
    |> Enum.join("\n\n")
  end

  defp render_user_block(%Terra.Document{} = doc), do: render_document_text(doc)
  defp render_user_block(%{type: "text", text: t}), do: t
  defp render_user_block(%{"type" => "text", "text" => t}), do: t
  defp render_user_block(_), do: ""

  defp stringify_tool_content(content) when is_binary(content), do: content

  defp stringify_tool_content(content) when is_list(content) do
    content
    |> Enum.map(fn
      %{type: "text", text: t} -> t
      %{"type" => "text", "text" => t} -> t
      %Terra.Document{content: c} -> c
      other -> inspect(other)
    end)
    |> Enum.join("\n")
  end

  defp stringify_tool_content(nil), do: ""
  defp stringify_tool_content(other), do: to_string(other)

  defp render_document_text(%Terra.Document{title: title, context: context, content: content}) do
    ctx_attr = if context && context != "", do: ~s| context="#{context}"|, else: ""
    ~s|<document title="#{title}"#{ctx_attr}>\n#{content}\n</document>|
  end

  # ── Streaming ──────────────────────────────────────────

  defp stream_handler(caller, ref) do
    fn {:data, data}, {req, res} ->
      if res.status not in 200..299 do
        existing = Process.get(:sse_error_body, "")
        Process.put(:sse_error_body, existing <> data)
        {:cont, {req, res}}
      else
        buffer = Process.get(:sse_buffer, "")
        combined = buffer <> data

        {lines, remaining} = extract_data_lines(combined)
        Process.put(:sse_buffer, remaining)

        state = Process.get(:openai_stream_state, initial_state())

        state =
          Enum.reduce(lines, state, fn line, st ->
            process_data_line(caller, ref, line, st)
          end)

        Process.put(:openai_stream_state, state)
        {:cont, {req, res}}
      end
    end
  end

  # Drain any remaining buffered SSE data when the stream completes. Without
  # this, a chunk that arrives without a trailing `\n\n` is silently dropped.
  defp flush_buffer(caller, ref) do
    remaining = Process.get(:sse_buffer, "")
    Process.delete(:sse_buffer)
    state = Process.get(:openai_stream_state, initial_state())
    Process.delete(:openai_stream_state)

    if remaining != "" do
      lines =
        remaining
        |> String.split("\n\n", trim: true)
        |> Enum.map(&extract_data_value/1)
        |> Enum.reject(&is_nil/1)

      Enum.reduce(lines, state, fn line, st ->
        process_data_line(caller, ref, line, st)
      end)
    end

    :ok
  end

  # ── State Machine ──────────────────────────────────────
  # Translates OpenAI's flat chunk format into Terra's structured event protocol.

  defp initial_state do
    %{
      message_started: false,
      text_block_started: false,
      # %{index => %{id, name, arguments}} — active tool calls
      tool_calls: %{},
      input_tokens: 0,
      output_tokens: 0
    }
  end

  defp process_data_line(_caller, _ref, "[DONE]", state) do
    state
  end

  defp process_data_line(caller, ref, json, state) do
    case Jason.decode(json) do
      {:ok, chunk} ->
        process_chunk(caller, ref, chunk, state)

      _ ->
        state
    end
  end

  defp process_chunk(caller, ref, %{"error" => err}, state) do
    error = %APIError{
      type: :stream_error,
      message: Map.get(err, "message", "stream error")
    }
    send(caller, {:stream, ref, {:error, error}})
    state
  end

  defp process_chunk(caller, ref, chunk, state) do
    choices = Map.get(chunk, "choices", [])
    usage = Map.get(chunk, "usage")
    message_id = Map.get(chunk, "id", "")

    # Handle usage-only chunk (final chunk, empty choices)
    state =
      if usage do
        input = Map.get(usage, "prompt_tokens", 0)
        output = Map.get(usage, "completion_tokens", 0)
        %{state | input_tokens: input, output_tokens: output}
      else
        state
      end

    # Process each choice
    state =
      Enum.reduce(choices, state, fn choice, st ->
        delta = Map.get(choice, "delta", %{})
        finish_reason = Map.get(choice, "finish_reason")

        st = ensure_message_started(caller, ref, message_id, st)
        st = process_delta(caller, ref, delta, st)
        st = process_finish(caller, ref, finish_reason, st)
        st
      end)

    # If this was usage-only (no choices) and we had choices before, send stop
    if choices == [] and usage != nil and state.message_started do
      stop_reason = Map.get(state, :pending_stop_reason, "end_turn")
      send(caller, {:stream, ref,
        {:message_delta,
         %{stop_reason: stop_reason},
         %{output_tokens: state.output_tokens}}})
      send(caller, {:stream, ref, :message_stop})
      state
    else
      state
    end
  end

  defp ensure_message_started(caller, ref, message_id, %{message_started: false} = state) do
    send(caller, {:stream, ref,
      {:message_start, %{
        id: message_id,
        content: [],
        usage: %{input_tokens: 0, output_tokens: 0}
      }}})

    %{state | message_started: true}
  end

  defp ensure_message_started(_, _, _, state), do: state

  defp process_delta(caller, ref, delta, state) do
    state = process_text_content(caller, ref, delta, state)
    state = process_tool_calls(caller, ref, delta, state)
    state
  end

  defp process_text_content(caller, ref, %{"content" => content}, state)
       when is_binary(content) and content != "" do
    state =
      if not state.text_block_started do
        send(caller, {:stream, ref,
          {:content_block_start, 0, %{type: "text", text: ""}}})
        %{state | text_block_started: true}
      else
        state
      end

    send(caller, {:stream, ref,
      {:content_block_delta, 0, %{type: "text_delta", text: content}}})

    state
  end

  defp process_text_content(_, _, _, state), do: state

  defp process_tool_calls(caller, ref, %{"tool_calls" => tool_calls}, state)
       when is_list(tool_calls) do
    Enum.reduce(tool_calls, state, fn tc, st ->
      index = Map.get(tc, "index", 0)
      tc_id = Map.get(tc, "id")
      function = Map.get(tc, "function", %{})
      name = Map.get(function, "name")
      arguments = Map.get(function, "arguments", "")

      # Tool call start — first chunk has id and name
      st =
        if tc_id != nil and not Map.has_key?(st.tool_calls, index) do
          # Close text block if open
          st =
            if st.text_block_started do
              send(caller, {:stream, ref, {:content_block_stop, 0}})
              %{st | text_block_started: false}
            else
              st
            end

          # Block index: text is 0, tool calls start at 1+
          block_index = index + 1

          send(caller, {:stream, ref,
            {:content_block_start, block_index,
             %{type: "tool_use", id: tc_id, name: name}}})

          tool_calls = Map.put(st.tool_calls, index, %{
            id: tc_id,
            name: name,
            block_index: block_index
          })

          %{st | tool_calls: tool_calls}
        else
          st
        end

      # Tool call delta — arguments chunk
      if arguments != "" do
        case Map.get(st.tool_calls, index) do
          %{block_index: block_index} ->
            send(caller, {:stream, ref,
              {:content_block_delta, block_index,
               %{type: "input_json_delta", partial_json: arguments}}})

          _ ->
            :ok
        end
      end

      st
    end)
  end

  defp process_tool_calls(_, _, _, state), do: state

  defp process_finish(caller, ref, finish_reason, state)
       when finish_reason != nil do
    had_tool_calls = map_size(state.tool_calls) > 0

    # Close text block
    state =
      if state.text_block_started do
        send(caller, {:stream, ref, {:content_block_stop, 0}})
        %{state | text_block_started: false}
      else
        state
      end

    # Close all tool call blocks
    state =
      state.tool_calls
      |> Enum.sort_by(fn {index, _} -> index end)
      |> Enum.reduce(state, fn {_index, %{block_index: block_index}}, st ->
        send(caller, {:stream, ref, {:content_block_stop, block_index}})
        st
      end)
      |> Map.put(:tool_calls, %{})

    # OpenAI returns finish_reason: "stop" when tool_choice forces a specific
    # function, even though the response is a tool call. Detect emitted tool
    # blocks and override.
    stop_reason =
      if had_tool_calls do
        "tool_use"
      else
        translate_finish_reason(finish_reason)
      end

    # message_delta and message_stop will be sent when usage arrives
    # (usage-only chunk with empty choices)
    %{state | tool_calls: %{}}
    |> Map.put(:pending_stop_reason, stop_reason)
  end

  defp process_finish(_, _, nil, state), do: state

  defp translate_finish_reason("stop"), do: "end_turn"
  defp translate_finish_reason("tool_calls"), do: "tool_use"
  defp translate_finish_reason("length"), do: "max_tokens"
  defp translate_finish_reason("content_filter"), do: "content_filter"
  defp translate_finish_reason(other), do: other

  # ── SSE Parsing ────────────────────────────────────────
  # OpenAI uses simple `data: {json}\n\n` format, no event: field.

  defp extract_data_lines(data) do
    case :binary.matches(data, "\n\n") do
      [] ->
        {[], data}

      matches ->
        {last_pos, _len} = List.last(matches)
        complete_end = last_pos + 2

        complete_part = binary_part(data, 0, complete_end)
        remaining = binary_part(data, complete_end, byte_size(data) - complete_end)

        lines =
          complete_part
          |> String.split("\n\n", trim: true)
          |> Enum.map(&extract_data_value/1)
          |> Enum.reject(&is_nil/1)

        {lines, remaining}
    end
  end

  defp extract_data_value(block) do
    block
    |> String.split("\n")
    |> Enum.find_value(fn line ->
      case line do
        "data: " <> value -> value
        _ -> nil
      end
    end)
  end
end
