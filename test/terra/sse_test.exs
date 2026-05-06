defmodule Terra.SSETest do
  use ExUnit.Case, async: true

  alias Terra.SSE

  describe "parse_chunk/2" do
    test "parses a complete event" do
      chunk = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}\n\n"

      {events, buffer} = SSE.parse_chunk(chunk, "")

      assert [%{event: "message_start", data: %{"type" => "message_start", "message" => %{"id" => "msg_1"}}}] = events
      assert buffer == ""
    end

    test "parses multiple events in one chunk" do
      chunk = """
      event: content_block_start
      data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

      event: content_block_delta
      data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

      """

      {events, buffer} = SSE.parse_chunk(chunk, "")

      assert length(events) == 2
      assert Enum.at(events, 0).event == "content_block_start"
      assert Enum.at(events, 1).event == "content_block_delta"
    end

    test "buffers incomplete events across chunks" do
      chunk1 = "event: message_start\ndata: {\"type\":\"mess"
      chunk2 = "age_start\",\"message\":{\"id\":\"msg_1\"}}\n\n"

      {events1, buffer} = SSE.parse_chunk(chunk1, "")
      assert events1 == []
      assert buffer != ""

      {events2, buffer2} = SSE.parse_chunk(chunk2, buffer)
      assert [%{event: "message_start"}] = events2
      assert buffer2 == ""
    end

    test "handles ping events" do
      chunk = "event: ping\ndata: {}\n\n"

      {events, _} = SSE.parse_chunk(chunk, "")
      assert [%{event: "ping"}] = events
    end

    test "handles message_stop (no data)" do
      chunk = "event: message_stop\ndata: {}\n\n"

      {events, _} = SSE.parse_chunk(chunk, "")
      assert [%{event: "message_stop"}] = events
    end

    test "handles error events" do
      chunk = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n"

      {events, _} = SSE.parse_chunk(chunk, "")
      assert [%{event: "error", data: %{"type" => "error"}}] = events
    end

    test "ignores empty lines and whitespace between events" do
      chunk = "\n\nevent: ping\ndata: {}\n\n\n\n"

      {events, _} = SSE.parse_chunk(chunk, "")
      assert [%{event: "ping"}] = events
    end
  end

  describe "to_terra_event/1" do
    test "message_start" do
      sse = %{event: "message_start", data: %{
        "type" => "message_start",
        "message" => %{
          "id" => "msg_1",
          "type" => "message",
          "role" => "assistant",
          "content" => [],
          "usage" => %{"input_tokens" => 25, "output_tokens" => 1}
        }
      }}

      assert {:message_start, msg} = SSE.to_terra_event(sse)
      assert msg.id == "msg_1"
      assert msg.usage.input_tokens == 25
    end

    test "content_block_start text" do
      sse = %{event: "content_block_start", data: %{
        "type" => "content_block_start",
        "index" => 0,
        "content_block" => %{"type" => "text", "text" => ""}
      }}

      assert {:content_block_start, 0, %{type: "text", text: ""}} = SSE.to_terra_event(sse)
    end

    test "content_block_start tool_use" do
      sse = %{event: "content_block_start", data: %{
        "type" => "content_block_start",
        "index" => 1,
        "content_block" => %{"type" => "tool_use", "id" => "toolu_01A", "name" => "get_weather"}
      }}

      assert {:content_block_start, 1, block} = SSE.to_terra_event(sse)
      assert block.type == "tool_use"
      assert block.id == "toolu_01A"
      assert block.name == "get_weather"
    end

    test "content_block_start thinking" do
      sse = %{event: "content_block_start", data: %{
        "type" => "content_block_start",
        "index" => 0,
        "content_block" => %{"type" => "thinking", "thinking" => ""}
      }}

      assert {:content_block_start, 0, %{type: "thinking", thinking: ""}} = SSE.to_terra_event(sse)
    end

    test "content_block_delta text_delta" do
      sse = %{event: "content_block_delta", data: %{
        "type" => "content_block_delta",
        "index" => 0,
        "delta" => %{"type" => "text_delta", "text" => "Hello"}
      }}

      assert {:content_block_delta, 0, %{type: "text_delta", text: "Hello"}} = SSE.to_terra_event(sse)
    end

    test "content_block_delta input_json_delta" do
      sse = %{event: "content_block_delta", data: %{
        "type" => "content_block_delta",
        "index" => 1,
        "delta" => %{"type" => "input_json_delta", "partial_json" => "{\"location\":"}
      }}

      assert {:content_block_delta, 1, %{type: "input_json_delta", partial_json: "{\"location\":"}} =
               SSE.to_terra_event(sse)
    end

    test "content_block_delta thinking_delta" do
      sse = %{event: "content_block_delta", data: %{
        "type" => "content_block_delta",
        "index" => 0,
        "delta" => %{"type" => "thinking_delta", "thinking" => "Let me think"}
      }}

      assert {:content_block_delta, 0, %{type: "thinking_delta", thinking: "Let me think"}} =
               SSE.to_terra_event(sse)
    end

    test "content_block_delta signature_delta" do
      sse = %{event: "content_block_delta", data: %{
        "type" => "content_block_delta",
        "index" => 0,
        "delta" => %{"type" => "signature_delta", "signature" => "EqQB..."}
      }}

      assert {:content_block_delta, 0, %{type: "signature_delta", signature: "EqQB..."}} =
               SSE.to_terra_event(sse)
    end

    test "content_block_stop" do
      sse = %{event: "content_block_stop", data: %{
        "type" => "content_block_stop",
        "index" => 0
      }}

      assert {:content_block_stop, 0} = SSE.to_terra_event(sse)
    end

    test "message_delta" do
      sse = %{event: "message_delta", data: %{
        "type" => "message_delta",
        "delta" => %{"stop_reason" => "end_turn"},
        "usage" => %{"output_tokens" => 15}
      }}

      assert {:message_delta, delta, usage} = SSE.to_terra_event(sse)
      assert delta.stop_reason == "end_turn"
      assert usage.output_tokens == 15
    end

    test "message_stop" do
      sse = %{event: "message_stop", data: %{}}

      assert :message_stop = SSE.to_terra_event(sse)
    end

    test "ping" do
      sse = %{event: "ping", data: %{}}

      assert :ping = SSE.to_terra_event(sse)
    end

    test "error" do
      sse = %{event: "error", data: %{
        "type" => "error",
        "error" => %{"type" => "overloaded_error", "message" => "Overloaded"}
      }}

      assert {:error, error} = SSE.to_terra_event(sse)
      assert error.type == "overloaded_error"
      assert error.message == "Overloaded"
    end
  end
end
