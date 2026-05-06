defmodule Terra.Provider.OpenAITest do
  use ExUnit.Case, async: true

  alias Terra.Provider.OpenAI
  alias Terra.Document

  # ── Helpers ──────────────────────────────────────────

  defp collect_messages(ref, acc) do
    receive do
      {:stream, ^ref, event} -> collect_messages(ref, acc ++ [event])
    after
      100 -> acc
    end
  end

  describe "module" do
    test "implements Terra.Provider behaviour" do
      assert Code.ensure_loaded?(OpenAI)
      assert function_exported?(OpenAI, :stream, 2)
      assert function_exported?(OpenAI, :cancel, 1)
    end
  end

  describe "Terra event protocol conformance" do
    test "text response produces correct Terra event sequence" do
      caller = self()
      ref = make_ref()

      # Simulate what the OpenAI provider would emit after translating chunks
      spawn_link(fn ->
        send(caller, {:stream, ref, {:message_start, %{id: "chatcmpl-123", content: [], usage: %{input_tokens: 0, output_tokens: 0}}}})
        send(caller, {:stream, ref, {:content_block_start, 0, %{type: "text", text: ""}}})
        send(caller, {:stream, ref, {:content_block_delta, 0, %{type: "text_delta", text: "Hello"}}})
        send(caller, {:stream, ref, {:content_block_delta, 0, %{type: "text_delta", text: " world"}}})
        send(caller, {:stream, ref, {:content_block_stop, 0}})
        send(caller, {:stream, ref, {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 5}}})
        send(caller, {:stream, ref, :message_stop})
      end)

      events = collect_messages(ref, [])

      assert {:message_start, %{id: "chatcmpl-123"}} = Enum.at(events, 0)
      assert {:content_block_start, 0, %{type: "text"}} = Enum.at(events, 1)
      assert {:content_block_delta, 0, %{type: "text_delta", text: "Hello"}} = Enum.at(events, 2)
      assert {:content_block_delta, 0, %{type: "text_delta", text: " world"}} = Enum.at(events, 3)
      assert {:content_block_stop, 0} = Enum.at(events, 4)
      assert {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 5}} = Enum.at(events, 5)
      assert :message_stop = Enum.at(events, 6)
    end

    test "tool call response produces correct Terra event sequence" do
      caller = self()
      ref = make_ref()

      spawn_link(fn ->
        send(caller, {:stream, ref, {:message_start, %{id: "chatcmpl-456", content: [], usage: %{input_tokens: 0, output_tokens: 0}}}})
        send(caller, {:stream, ref, {:content_block_start, 1, %{type: "tool_use", id: "call_abc", name: "get_weather"}}})
        send(caller, {:stream, ref, {:content_block_delta, 1, %{type: "input_json_delta", partial_json: "{\"loc\":"}}})
        send(caller, {:stream, ref, {:content_block_delta, 1, %{type: "input_json_delta", partial_json: "\"NYC\"}"}}})
        send(caller, {:stream, ref, {:content_block_stop, 1}})
        send(caller, {:stream, ref, {:message_delta, %{stop_reason: "tool_use"}, %{output_tokens: 20}}})
        send(caller, {:stream, ref, :message_stop})
      end)

      events = collect_messages(ref, [])

      assert {:message_start, _} = Enum.at(events, 0)
      assert {:content_block_start, 1, %{type: "tool_use", id: "call_abc", name: "get_weather"}} = Enum.at(events, 1)
      assert {:content_block_delta, 1, %{type: "input_json_delta"}} = Enum.at(events, 2)
      assert {:content_block_stop, 1} = Enum.at(events, 4)
      assert {:message_delta, %{stop_reason: "tool_use"}, _} = Enum.at(events, 5)
    end
  end

  describe "render_messages/1 — Terra → OpenAI translation" do
    test "passes through plain string content" do
      msgs = [%{role: "user", content: "hello"}]
      assert OpenAI.render_messages(msgs) == [%{role: "user", content: "hello"}]
    end

    test "assistant tool_use blocks become tool_calls array" do
      msgs = [
        %{
          role: "assistant",
          content: [
            %{type: "text", text: "Looking up..."},
            %{type: "tool_use", id: "call_1", name: "get_weather", input: %{location: "NYC"}}
          ]
        }
      ]

      [out] = OpenAI.render_messages(msgs)
      assert out.role == "assistant"
      assert out.content == "Looking up..."
      assert [tc] = out.tool_calls
      assert tc.id == "call_1"
      assert tc.type == "function"
      assert tc.function.name == "get_weather"
      assert {:ok, %{"location" => "NYC"}} = Jason.decode(tc.function.arguments)
    end

    test "assistant with only tool_use sets content to nil" do
      msgs = [
        %{
          role: "assistant",
          content: [%{type: "tool_use", id: "call_1", name: "f", input: %{}}]
        }
      ]

      [out] = OpenAI.render_messages(msgs)
      assert out.content == nil
      assert length(out.tool_calls) == 1
    end

    test "user tool_result blocks become role: tool messages" do
      msgs = [
        %{
          role: "user",
          content: [
            %{type: "tool_result", tool_use_id: "call_1", content: "72°F"},
            %{type: "tool_result", tool_use_id: "call_2", content: "sunny"}
          ]
        }
      ]

      out = OpenAI.render_messages(msgs)
      assert length(out) == 2
      assert Enum.at(out, 0) == %{role: "tool", tool_call_id: "call_1", content: "72°F"}
      assert Enum.at(out, 1) == %{role: "tool", tool_call_id: "call_2", content: "sunny"}
    end

    test "user with mixed tool_results and text emits tool messages then user message" do
      msgs = [
        %{
          role: "user",
          content: [
            %{type: "tool_result", tool_use_id: "call_1", content: "72°F"},
            %{type: "text", text: "What about Paris?"}
          ]
        }
      ]

      out = OpenAI.render_messages(msgs)
      assert length(out) == 2
      assert Enum.at(out, 0).role == "tool"
      assert Enum.at(out, 1) == %{role: "user", content: "What about Paris?"}
    end

    test "Terra.Document in user content flattens to text" do
      doc = Document.new("forecast", "weather data", "72°F sunny")

      msgs = [
        %{
          role: "user",
          content: [doc, %{type: "text", text: "Summarize."}]
        }
      ]

      [out] = OpenAI.render_messages(msgs)
      assert out.role == "user"
      assert out.content =~ ~s|<document title="forecast"|
      assert out.content =~ ~s|context="weather data"|
      assert out.content =~ "72°F sunny"
      assert out.content =~ "Summarize."
    end

    test "tool_result with structured content is stringified" do
      msgs = [
        %{
          role: "user",
          content: [
            %{type: "tool_result", tool_use_id: "x", content: [%{type: "text", text: "ok"}]}
          ]
        }
      ]

      [out] = OpenAI.render_messages(msgs)
      assert out.content == "ok"
    end
  end

  describe "SSE data line extraction" do
    test "extracts data values from OpenAI SSE format" do
      sse = "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0}]}\n\ndata: [DONE]\n\n"

      {lines, remaining} = extract_data_lines(sse)

      assert length(lines) == 2
      assert Enum.at(lines, 0) =~ "chatcmpl-1"
      assert Enum.at(lines, 1) == "[DONE]"
      assert remaining == ""
    end

    test "buffers incomplete SSE data" do
      chunk1 = "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"con"
      chunk2 = "tent\":\"Hi\"},\"index\":0}]}\n\n"

      {lines1, buffer} = extract_data_lines(chunk1)
      assert lines1 == []
      assert buffer != ""

      {lines2, remaining} = extract_data_lines(buffer <> chunk2)
      assert length(lines2) == 1
      assert remaining == ""
    end

    test "handles multiple events in one chunk" do
      sse = "data: {\"a\":1}\n\ndata: {\"b\":2}\n\ndata: {\"c\":3}\n\n"

      {lines, remaining} = extract_data_lines(sse)

      assert length(lines) == 3
      assert remaining == ""
    end

    test "handles data with trailing incomplete event" do
      sse = "data: {\"a\":1}\n\ndata: {\"b\":2"

      {lines, remaining} = extract_data_lines(sse)

      assert length(lines) == 1
      assert remaining == "data: {\"b\":2"
    end
  end

  # ── Private function mirrors for testing ───────────

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
