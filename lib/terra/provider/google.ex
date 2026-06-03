defmodule Terra.Provider.Google do
  @moduledoc """
  Google Gemini streaming provider.

  Translates Gemini's parts-based streaming format into Terra's event protocol.
  Gemini uses SSE via `?alt=sse` with `data: {json}\\n\\n` chunks. Unlike
  Anthropic/OpenAI, each chunk contains full candidate parts (not deltas).

  ## Config

      def context(_state_name, _state) do
        Terra.Context.new()
        |> Terra.Context.system("You are helpful.")
        |> Terra.Context.model(%{
          model: "gemini-2.5-flash",
          max_tokens: 4096,
          config: %{
            api_key: System.get_env("GOOGLE_API_KEY"),
            base_url: "https://generativelanguage.googleapis.com/v1beta",  # optional
            receive_timeout: 120_000,                                       # optional
            finch_name: MyApp.Finch                                         # optional
          }
        })
        |> Terra.Context.build()
      end

  ## Supported Parameters

  - `:model` — model ID (required)
  - `:max_tokens` — maps to `maxOutputTokens` in generation config
  - `:system` — system prompt (sent as `systemInstruction`)
  - `:messages` — conversation history (role `"assistant"` mapped to `"model"`)
  - `:tools` — tool definitions (formatted as `functionDeclarations`)
  - `:thinking` — maps to `thinkingConfig` in generation config
  - `:temperature`, `:top_p`, `:top_k`, `:stop_sequences`

  ## Translation

  | Gemini Part                            | Terra Event                              |
  |----------------------------------------|------------------------------------------|
  | `%{"text" => "..."}`                   | `{:content_block_delta, idx, text_delta}` |
  | `%{"thought" => true, "text" => ...}`  | `{:content_block_delta, idx, thinking_delta}` |
  | `%{"functionCall" => ...}`             | Start + delta + stop (complete in one go) |
  | `finishReason: "STOP"`                 | `stop_reason: "end_turn"`                 |
  | `finishReason: "MAX_TOKENS"`           | `stop_reason: "max_tokens"`               |
  | `usageMetadata`                        | Token tracking (prompt + candidates + thoughts) |
  """

  @behaviour Terra.Provider

  alias Terra.APIError

  @default_base_url "https://generativelanguage.googleapis.com/v1beta"
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
    model = Map.fetch!(params, :model)

    body = build_body(params)
    url = "#{base_url}/models/#{model}:streamGenerateContent?alt=sse&key=#{api_key}"
    headers = %{"content-type" => "application/json"}

    finch_opts =
      case Map.get(config, :finch_name) do
        nil -> []
        name -> [finch: name]
      end

    Terra.Telemetry.provider_request(caller, body, headers)

    req =
      Req.new(
        [
          url: url,
          method: :post,
          headers: headers,
          body: Jason.encode!(body),
          receive_timeout: timeout,
          into: stream_handler(caller, ref)
        ] ++ finch_opts
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

  defp build_body(params) do
    body = %{}

    # Generation config
    gen_config =
      %{}
      |> maybe_put(:maxOutputTokens, Map.get(params, :max_tokens))
      |> maybe_put(:temperature, Map.get(params, :temperature))
      |> maybe_put(:topP, Map.get(params, :top_p))
      |> maybe_put(:topK, Map.get(params, :top_k))
      |> maybe_put(:stopSequences, Map.get(params, :stop_sequences))

    gen_config =
      case Map.get(params, :thinking) do
        nil -> gen_config
        thinking -> Map.put(gen_config, :thinkingConfig, thinking)
      end

    body = if gen_config != %{}, do: Map.put(body, :generationConfig, gen_config), else: body

    # System instruction
    body =
      case Map.get(params, :system) do
        nil -> body
        "" -> body
        system -> Map.put(body, :systemInstruction, %{parts: [%{text: system}]})
      end

    # Contents (messages)
    body =
      case Map.get(params, :messages, []) do
        [] -> body
        messages -> Map.put(body, :contents, format_messages(messages))
      end

    # Tools
    case Map.get(params, :tools, []) do
      [] ->
        body

      tools ->
        formatted =
          Enum.map(tools, fn tool ->
            %{
              name: tool.name,
              description: tool.description,
              parameters: tool.input_schema
            }
          end)

        Map.put(body, :tools, [%{functionDeclarations: formatted}])
    end
  end

  # Terra agents emit Anthropic-flavored content blocks. Translate to Gemini's
  # parts format. tool_result blocks reference `tool_use_id` (not name), so we
  # build an id→name index from prior assistant tool_use blocks.
  @doc false
  def format_messages(messages) do
    tool_index = build_tool_use_index(messages)

    messages
    |> Enum.flat_map(fn msg ->
      role =
        case msg["role"] || msg[:role] do
          "assistant" -> "model"
          "user" -> "user"
          other -> other
        end

      content = msg["content"] || msg[:content]

      parts =
        cond do
          is_binary(content) -> [%{text: content}]
          is_list(content) -> Enum.flat_map(content, &format_content_part(&1, tool_index))
          true -> [%{text: to_string(content)}]
        end

      case parts do
        [] -> []
        _ -> [%{role: role, parts: parts}]
      end
    end)
  end

  defp build_tool_use_index(messages) do
    Enum.reduce(messages, %{}, fn msg, acc ->
      role = msg["role"] || msg[:role]
      content = msg["content"] || msg[:content]

      if role == "assistant" and is_list(content) do
        Enum.reduce(content, acc, fn
          %{type: "tool_use", id: id, name: name}, a -> Map.put(a, id, name)
          %{"type" => "tool_use", "id" => id, "name" => name}, a -> Map.put(a, id, name)
          _, a -> a
        end)
      else
        acc
      end
    end)
  end

  defp format_content_part(%Terra.Document{} = doc, _) do
    [%{text: render_document_text(doc)}]
  end

  defp format_content_part(%{type: "text", text: text}, _), do: [%{text: text}]
  defp format_content_part(%{"type" => "text", "text" => text}, _), do: [%{text: text}]

  defp format_content_part(%{type: "tool_use", name: name, input: input}, _) do
    [%{functionCall: %{name: name, args: input || %{}}}]
  end

  defp format_content_part(%{"type" => "tool_use", "name" => name, "input" => input}, _) do
    [%{functionCall: %{name: name, args: input || %{}}}]
  end

  defp format_content_part(%{type: "tool_result", tool_use_id: id, content: content}, tool_index) do
    name = Map.get(tool_index, id, id)
    [%{functionResponse: %{name: name, response: tool_response(content)}}]
  end

  defp format_content_part(%{"type" => "tool_result", "tool_use_id" => id, "content" => content}, tool_index) do
    name = Map.get(tool_index, id, id)
    [%{functionResponse: %{name: name, response: tool_response(content)}}]
  end

  # Gemini doesn't accept thinking parts as input — drop them from history.
  defp format_content_part(%{type: "thinking"}, _), do: []
  defp format_content_part(%{"type" => "thinking"}, _), do: []
  defp format_content_part(%{type: "redacted_thinking"}, _), do: []
  defp format_content_part(%{"type" => "redacted_thinking"}, _), do: []

  defp format_content_part(other, _), do: [%{text: inspect(other)}]

  # functionResponse.response must be a JSON object — wrap scalars under :result.
  defp tool_response(content) when is_binary(content), do: %{result: content}
  defp tool_response(content) when is_map(content) and not is_struct(content), do: content

  defp tool_response(content) when is_list(content) do
    text =
      content
      |> Enum.map(fn
        %{type: "text", text: t} -> t
        %{"type" => "text", "text" => t} -> t
        %Terra.Document{content: c} -> c
        other -> inspect(other)
      end)
      |> Enum.join("\n")

    %{result: text}
  end

  defp tool_response(nil), do: %{result: ""}
  defp tool_response(other), do: %{result: to_string(other)}

  defp render_document_text(%Terra.Document{title: title, context: context, content: content}) do
    ctx_attr = if context && context != "", do: ~s| context="#{context}"|, else: ""
    ~s|<document title="#{title}"#{ctx_attr}>\n#{content}\n</document>|
  end

  defp maybe_put(map, _key, nil), do: map
  defp maybe_put(map, _key, []), do: map
  defp maybe_put(map, key, value), do: Map.put(map, key, value)

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

        {chunks, remaining} = extract_data_lines(combined)
        Process.put(:sse_buffer, remaining)

        state = Process.get(:gemini_stream_state, initial_state())

        state =
          Enum.reduce(chunks, state, fn json, st ->
            case Jason.decode(json) do
              {:ok, chunk} -> process_chunk(caller, ref, chunk, st)
              _ -> st
            end
          end)

        Process.put(:gemini_stream_state, state)
        {:cont, {req, res}}
      end
    end
  end

  # Drain any remaining buffered SSE data when the stream completes. Without
  # this, a small response that arrives in a single chunk lacking a trailing
  # `\n\n` is silently dropped (extract_data_lines requires the terminator).
  defp flush_buffer(caller, ref) do
    remaining = Process.get(:sse_buffer, "")
    Process.delete(:sse_buffer)
    state = Process.get(:gemini_stream_state, initial_state())
    Process.delete(:gemini_stream_state)

    state =
      if remaining != "" do
        chunks =
          remaining
          |> String.split("\n\n", trim: true)
          |> Enum.map(&extract_data_value/1)
          |> Enum.reject(&is_nil/1)
          |> Enum.reject(&(&1 == "[DONE]"))

        Enum.reduce(chunks, state, fn json, st ->
          case Jason.decode(json) do
            {:ok, chunk} -> process_chunk(caller, ref, chunk, st)
            _ -> st
          end
        end)
      else
        state
      end

    # Some Gemini responses close the stream without a final chunk carrying
    # `finishReason`. Synthesize a terminal message_delta + message_stop so
    # downstream agents receive a complete event sequence.
    if state.message_started and not state.message_completed do
      state = close_open_blocks(caller, ref, state)

      stop_reason =
        if state.tool_use_emitted, do: "tool_use", else: "end_turn"

      send(caller, {:stream, ref,
        {:message_delta, %{stop_reason: stop_reason},
         %{output_tokens: state.output_tokens}}})
      send(caller, {:stream, ref, :message_stop})
    end

    :ok
  end

  # ── State Machine ──────────────────────────────────────

  defp initial_state do
    %{
      message_started: false,
      message_completed: false,
      tool_use_emitted: false,
      block_counter: 0,
      text_block_open: false,
      thinking_block_open: false,
      input_tokens: 0,
      output_tokens: 0
    }
  end

  defp process_chunk(caller, ref, %{"error" => err}, state) do
    error = %APIError{
      type: :stream_error,
      status: Map.get(err, "code"),
      message: Map.get(err, "message", "stream error")
    }
    send(caller, {:stream, ref, {:error, error}})
    state
  end

  defp process_chunk(caller, ref, %{"promptFeedback" => %{"blockReason" => reason} = pf}, state) do
    error = %APIError{
      type: :stream_error,
      message: "prompt blocked: #{reason}#{format_safety(pf)}"
    }
    send(caller, {:stream, ref, {:error, error}})
    state
  end

  defp process_chunk(caller, ref, chunk, state) do
    candidates = Map.get(chunk, "candidates", [])
    usage = Map.get(chunk, "usageMetadata")

    state =
      if usage do
        input = Map.get(usage, "promptTokenCount", 0)
        output = Map.get(usage, "candidatesTokenCount", 0) + Map.get(usage, "thoughtsTokenCount", 0)
        %{state | input_tokens: input, output_tokens: output}
      else
        state
      end

    Enum.reduce(candidates, state, fn candidate, st ->
      process_candidate(caller, ref, candidate, st)
    end)
  end

  defp process_candidate(caller, ref, candidate, state) do
    content = get_in(candidate, ["content", "parts"]) || []
    finish_reason = Map.get(candidate, "finishReason")

    state = ensure_message_started(caller, ref, state)

    state =
      Enum.reduce(content, state, fn part, st ->
        process_part(caller, ref, part, st)
      end)

    if finish_reason do
      state = close_open_blocks(caller, ref, state)

      # Gemini returns finishReason: "STOP" even when a functionCall part was
      # emitted. Detect tool emissions and override so callers can dispatch
      # tools correctly.
      stop_reason =
        if state.tool_use_emitted do
          "tool_use"
        else
          translate_finish_reason(finish_reason)
        end

      send(caller, {:stream, ref,
        {:message_delta, %{stop_reason: stop_reason},
         %{output_tokens: state.output_tokens}}})
      send(caller, {:stream, ref, :message_stop})

      %{state | message_completed: true}
    else
      state
    end
  end

  defp ensure_message_started(caller, ref, %{message_started: false} = state) do
    send(caller, {:stream, ref,
      {:message_start, %{
        id: "gemini-#{System.unique_integer([:positive])}",
        content: [],
        usage: %{input_tokens: 0, output_tokens: 0}
      }}})
    %{state | message_started: true}
  end

  defp ensure_message_started(_, _, state), do: state

  defp process_part(caller, ref, %{"thought" => true, "text" => text}, state) when is_binary(text) do
    # Close text block if open
    state = if state.text_block_open, do: close_text_block(caller, ref, state), else: state

    state =
      if not state.thinking_block_open do
        idx = state.block_counter
        send(caller, {:stream, ref, {:content_block_start, idx, %{type: "thinking", thinking: ""}}})
        %{state | thinking_block_open: true, block_counter: idx}
      else
        state
      end

    send(caller, {:stream, ref,
      {:content_block_delta, state.block_counter,
       %{type: "thinking_delta", thinking: text}}})

    state
  end

  defp process_part(caller, ref, %{"text" => text}, state) when is_binary(text) do
    # Close thinking block if open
    state = if state.thinking_block_open, do: close_thinking_block(caller, ref, state), else: state

    state =
      if not state.text_block_open do
        idx = state.block_counter
        send(caller, {:stream, ref, {:content_block_start, idx, %{type: "text", text: ""}}})
        %{state | text_block_open: true, block_counter: idx}
      else
        state
      end

    send(caller, {:stream, ref,
      {:content_block_delta, state.block_counter,
       %{type: "text_delta", text: text}}})

    state
  end

  defp process_part(caller, ref, %{"functionCall" => fc}, state) do
    state = close_open_blocks(caller, ref, state)

    idx = state.block_counter
    name = Map.get(fc, "name", "")
    args = Map.get(fc, "args", %{})
    tool_id = "tool_#{System.unique_integer([:positive])}"

    send(caller, {:stream, ref,
      {:content_block_start, idx,
       %{type: "tool_use", id: tool_id, name: name}}})

    json = Jason.encode!(args)
    send(caller, {:stream, ref,
      {:content_block_delta, idx,
       %{type: "input_json_delta", partial_json: json}}})

    send(caller, {:stream, ref, {:content_block_stop, idx}})

    %{state | block_counter: idx + 1, tool_use_emitted: true}
  end

  defp process_part(_, _, _, state), do: state

  defp close_text_block(caller, ref, state) do
    send(caller, {:stream, ref, {:content_block_stop, state.block_counter}})
    %{state | text_block_open: false, block_counter: state.block_counter + 1}
  end

  defp close_thinking_block(caller, ref, state) do
    send(caller, {:stream, ref, {:content_block_stop, state.block_counter}})
    %{state | thinking_block_open: false, block_counter: state.block_counter + 1}
  end

  defp close_open_blocks(caller, ref, state) do
    state = if state.text_block_open, do: close_text_block(caller, ref, state), else: state
    state = if state.thinking_block_open, do: close_thinking_block(caller, ref, state), else: state
    state
  end

  defp format_safety(%{"safetyRatings" => ratings}) when is_list(ratings) and ratings != [] do
    blocked =
      ratings
      |> Enum.filter(&Map.get(&1, "blocked"))
      |> Enum.map(&Map.get(&1, "category"))

    case blocked do
      [] -> ""
      cats -> " (#{Enum.join(cats, ", ")})"
    end
  end

  defp format_safety(_), do: ""

  defp translate_finish_reason("STOP"), do: "end_turn"
  defp translate_finish_reason("MAX_TOKENS"), do: "max_tokens"
  defp translate_finish_reason("SAFETY"), do: "content_filter"
  defp translate_finish_reason("RECITATION"), do: "content_filter"
  defp translate_finish_reason(other), do: other

  # ── SSE Parsing ────────────────────────────────────────

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
          |> Enum.reject(&(&1 == "[DONE]"))

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
