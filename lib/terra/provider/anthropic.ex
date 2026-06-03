defmodule Terra.Provider.Anthropic do
  @moduledoc """
  Anthropic Messages API streaming provider.

  Streams Claude responses via SSE and translates them into Terra's event
  protocol. Uses Req with callback-based `into:` streaming and process
  dictionary buffering for SSE chunks.

  ## Config

  Pass config in the `model` map returned by your agent's `context/2`:

      def context(_state_name, _state) do
        Terra.Context.new()
        |> Terra.Context.system("You are helpful.")
        |> Terra.Context.model(%{
          model: "claude-sonnet-4-5-20250929",
          max_tokens: 4096,
          config: %{
            api_key: System.get_env("ANTHROPIC_API_KEY"),
            base_url: "https://api.anthropic.com",            # optional
            beta: ["interleaved-thinking-2025-05-14"],         # optional
            receive_timeout: 120_000,                          # optional
            finch_name: MyApp.Finch                            # optional
          }
        })
        |> Terra.Context.build()
      end

  ## Supported Parameters

  - `:model` — model ID (required)
  - `:max_tokens` — max output tokens (default: 4096)
  - `:system` — system prompt
  - `:messages` — conversation history
  - `:tools` — tool definitions (with optional `:cache_control`)
  - `:thinking` — `%{type: "enabled", budget_tokens: n}` for extended thinking
  - `:tool_choice` — `%{type: "auto" | "any" | "tool", name: "..."}
  - `:temperature`, `:top_p`, `:top_k`, `:stop_sequences`, `:metadata`

  ## Error Handling

  Non-2xx responses are accumulated and parsed into `Terra.APIError` structs
  with status, type, message, request_id, and retry_after fields.
  SSE error events during streaming are also translated to `Terra.APIError`.
  """

  @behaviour Terra.Provider

  alias Terra.APIError

  @default_base_url "https://api.anthropic.com"
  @api_version "2023-06-01"
  @default_timeout 120_000

  @sse_events [
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
    "error"
  ]

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
    beta = Map.get(config, :beta, [])

    body = build_body(params)
    headers = build_headers(api_key, beta)

    finch_opts =
      case Map.get(config, :finch_name) do
        nil -> []
        name -> [finch: name]
      end

    Terra.Telemetry.provider_request(caller, body, headers)

    req =
      Req.new(
        [
          base_url: base_url,
          url: "/v1/messages",
          method: :post,
          headers: headers,
          body: Jason.encode!(body),
          receive_timeout: timeout,
          into: stream_handler(caller, ref)
        ] ++ finch_opts
      )

    result = Req.request(req)

    case result do
      {:ok, %{status: status}} when status in 200..299 ->
        :ok

      {:ok, resp} ->
        # For non-2xx, the stream handler accumulated the error body
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

  defp build_headers(api_key, beta) do
    headers = %{
      "content-type" => "application/json",
      "x-api-key" => api_key,
      "anthropic-version" => @api_version
    }

    case beta do
      [] -> headers
      tokens when is_list(tokens) -> Map.put(headers, "anthropic-beta", Enum.join(tokens, ","))
      token when is_binary(token) -> Map.put(headers, "anthropic-beta", token)
    end
  end

  defp build_body(params) do
    body = %{
      model: Map.fetch!(params, :model),
      max_tokens: Map.get(params, :max_tokens, 4096),
      stream: true
    }

    optional_fields = [
      {:system, :system},
      {:messages, :messages},
      {:thinking, :thinking},
      {:tool_choice, :tool_choice},
      {:temperature, :temperature},
      {:top_p, :top_p},
      {:top_k, :top_k},
      {:stop_sequences, :stop_sequences},
      {:metadata, :metadata}
    ]

    body =
      Enum.reduce(optional_fields, body, fn {param_key, body_key}, acc ->
        case Map.get(params, param_key) do
          nil -> acc
          "" -> acc
          [] -> acc
          value -> Map.put(acc, body_key, value)
        end
      end)

    # Render Terra.Document blocks in messages to Anthropic API format
    body =
      case Map.get(body, :messages) do
        nil -> body
        messages -> Map.put(body, :messages, render_messages(messages))
      end

    case Map.get(params, :tools, []) do
      [] -> body
      tools -> Map.put(body, :tools, format_tools(tools))
    end
  end

  defp format_tools(tools) do
    Enum.map(tools, fn tool ->
      base = %{
        name: tool.name,
        description: tool.description,
        input_schema: tool.input_schema
      }

      case Map.get(tool, :cache_control) do
        nil -> base
        cc -> Map.put(base, :cache_control, cc)
      end
    end)
  end

  defp render_messages(messages) do
    Enum.map(messages, fn
      %{content: content} = msg when is_list(content) ->
        %{msg | content: Enum.map(content, &render_content_block/1)}

      msg ->
        msg
    end)
  end

  defp render_content_block(%{type: "tool_result", content: content} = block) when is_list(content) do
    %{block | content: Enum.map(content, &render_content_block/1)}
  end

  defp render_content_block(%{type: "tool_result", content: %Terra.Document{} = doc} = block) do
    %{block | content: [render_content_block(doc)]}
  end

  defp render_content_block(%Terra.Document{} = doc) do
    block = %{
      type: "document",
      source: %{type: "text", media_type: "text/plain", data: doc.content},
      title: doc.title,
      context: doc.context,
      citations: %{enabled: false}
    }

    case doc.cache do
      :ephemeral -> Map.put(block, :cache_control, %{type: "ephemeral"})
      _ -> block
    end
  end

  defp render_content_block(block), do: block

  # ── Streaming ──────────────────────────────────────────

  # Callback-based stream handler — runs inside the Req process.
  # Uses process dictionary for SSE buffering (same pattern as anthropix).
  defp stream_handler(caller, ref) do
    fn {:data, data}, {req, res} ->
      if res.status not in 200..299 do
        # Non-2xx: accumulate raw body for error parsing later
        existing = Process.get(:sse_error_body, "")
        Process.put(:sse_error_body, existing <> data)
        {:cont, {req, res}}
      else
        buffer = Process.get(:sse_buffer, "")
        combined = buffer <> data

        {complete_events, remaining} = extract_complete_sse_events(combined)
        Process.put(:sse_buffer, remaining)

        Enum.each(complete_events, fn %{event: event_type} = sse ->
          if event_type in @sse_events do
            if event_type == "error" do
              error = APIError.from_sse_event(sse.data)
              send(caller, {:stream, ref, {:error, error}})
            else
              terra_event = Terra.SSE.to_terra_event(sse)
              send(caller, {:stream, ref, terra_event})
            end
          end
        end)

        {:cont, {req, res}}
      end
    end
  end

  # Extract complete SSE events from buffered data.
  # An event is complete when followed by a double newline.
  defp extract_complete_sse_events(data) do
    case :binary.matches(data, "\n\n") do
      [] ->
        {[], data}

      matches ->
        {last_pos, _len} = List.last(matches)
        complete_end = last_pos + 2

        complete_part = binary_part(data, 0, complete_end)
        remaining = binary_part(data, complete_end, byte_size(data) - complete_end)

        events =
          complete_part
          |> String.split("\n\n", trim: true)
          |> Enum.map(&parse_sse_block/1)
          |> Enum.reject(&is_nil/1)

        {events, remaining}
    end
  end

  defp parse_sse_block(block) do
    lines =
      block
      |> String.split("\n")
      |> Enum.reject(&(&1 == ""))

    fields =
      Enum.reduce(lines, %{}, fn line, acc ->
        case String.split(line, ": ", parts: 2) do
          [key, value] -> Map.put(acc, key, value)
          _ -> acc
        end
      end)

    case fields do
      %{"event" => event_type, "data" => json} ->
        case Jason.decode(json) do
          {:ok, data} -> %{event: event_type, data: data}
          _ -> nil
        end

      %{"event" => event_type} ->
        %{event: event_type, data: %{}}

      _ ->
        nil
    end
  end

end
