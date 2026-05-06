defmodule Terra.SSE do
  @moduledoc """
  SSE (Server-Sent Events) parser for Anthropic's streaming API.

  Handles raw SSE text with buffering for incomplete events and converts
  parsed events into Terra's internal event tuples.

  ## Usage

      # Parse a chunk, carrying over the buffer from the previous chunk
      {events, buffer} = Terra.SSE.parse_chunk(raw_data, buffer)

      # Convert each parsed SSE event into a Terra event tuple
      terra_events = Enum.map(events, &Terra.SSE.to_terra_event/1)

  ## SSE Format

  Anthropic SSE events have `event:` and `data:` fields separated by `\\n\\n`:

      event: message_start
      data: {"type":"message_start","message":{...}}

      event: content_block_delta
      data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}

  This module is used internally by `Terra.Provider.Anthropic`. OpenAI and
  Google providers handle their own SSE parsing inline since their formats differ.
  """

  @doc """
  Parses an SSE chunk, returning completed events and remaining buffer.
  """
  @spec parse_chunk(String.t(), String.t()) :: {[map()], String.t()}
  def parse_chunk(chunk, buffer) do
    raw = buffer <> chunk

    # Split on double newline (event boundary)
    case String.split(raw, "\n\n") do
      [incomplete] ->
        {[], incomplete}

      parts ->
        {complete, [remainder]} = Enum.split(parts, -1)

        events =
          complete
          |> Enum.map(&parse_event/1)
          |> Enum.reject(&is_nil/1)

        {events, remainder}
    end
  end

  defp parse_event(raw_event) do
    lines =
      raw_event
      |> String.split("\n")
      |> Enum.map(&String.trim/1)
      |> Enum.reject(&(&1 == ""))

    fields =
      Enum.reduce(lines, %{}, fn line, acc ->
        case String.split(line, ": ", parts: 2) do
          [key, value] -> Map.put(acc, key, value)
          _ -> acc
        end
      end)

    case fields do
      %{"event" => event_type} ->
        data =
          case Map.get(fields, "data") do
            nil -> %{}
            json -> Jason.decode!(json)
          end

        %{event: event_type, data: data}

      _ ->
        nil
    end
  end

  # ── to_terra_event ─────────────────────────────────────

  @doc """
  Converts a parsed SSE event map into a Terra event tuple.
  """
  def to_terra_event(%{event: "message_start", data: %{"message" => msg}}) do
    {:message_start, atomize_message(msg)}
  end

  def to_terra_event(%{event: "content_block_start", data: %{"index" => idx, "content_block" => block}}) do
    {:content_block_start, idx, atomize_block(block)}
  end

  def to_terra_event(%{event: "content_block_delta", data: %{"index" => idx, "delta" => delta}}) do
    {:content_block_delta, idx, atomize_delta(delta)}
  end

  def to_terra_event(%{event: "content_block_stop", data: %{"index" => idx}}) do
    {:content_block_stop, idx}
  end

  def to_terra_event(%{event: "message_delta", data: %{"delta" => delta, "usage" => usage}}) do
    {:message_delta, atomize_keys(delta), atomize_keys(usage)}
  end

  def to_terra_event(%{event: "message_stop"}) do
    :message_stop
  end

  def to_terra_event(%{event: "ping"}) do
    :ping
  end

  def to_terra_event(%{event: "error", data: %{"error" => error}}) do
    {:error, atomize_keys(error)}
  end

  # ── Helpers ─────────────────────────────────────────────

  defp atomize_message(msg) do
    usage = atomize_keys(msg["usage"] || %{})

    %{
      id: msg["id"],
      type: msg["type"],
      role: msg["role"],
      content: msg["content"],
      usage: usage
    }
  end

  defp atomize_block(%{"type" => "text"} = b), do: %{type: "text", text: b["text"]}
  defp atomize_block(%{"type" => "thinking"} = b), do: %{type: "thinking", thinking: b["thinking"]}

  defp atomize_block(%{"type" => "tool_use"} = b) do
    %{type: "tool_use", id: b["id"], name: b["name"]}
  end

  defp atomize_block(b), do: atomize_keys(b)

  defp atomize_delta(%{"type" => "text_delta"} = d), do: %{type: "text_delta", text: d["text"]}
  defp atomize_delta(%{"type" => "thinking_delta"} = d), do: %{type: "thinking_delta", thinking: d["thinking"]}
  defp atomize_delta(%{"type" => "signature_delta"} = d), do: %{type: "signature_delta", signature: d["signature"]}

  defp atomize_delta(%{"type" => "input_json_delta"} = d) do
    %{type: "input_json_delta", partial_json: d["partial_json"]}
  end

  defp atomize_delta(d), do: atomize_keys(d)

  defp atomize_keys(map) when is_map(map) do
    Map.new(map, fn {k, v} -> {String.to_atom(k), v} end)
  end
end
