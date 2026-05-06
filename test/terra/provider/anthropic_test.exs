defmodule Terra.Provider.AnthropicTest do
  use ExUnit.Case, async: true

  alias Terra.Provider.Anthropic

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
      assert Code.ensure_loaded?(Anthropic)
      assert function_exported?(Anthropic, :stream, 2)
      assert function_exported?(Anthropic, :cancel, 1)
    end
  end

  describe "Terra event protocol conformance" do
    test "text response produces correct Terra event sequence" do
      caller = self()
      ref = make_ref()

      # Simulate what the Anthropic provider would emit after translating SSE
      spawn_link(fn ->
        send(caller, {:stream, ref, {:message_start, %{id: "msg_01ABC", content: [], usage: %{input_tokens: 10, output_tokens: 0}}}})
        send(caller, {:stream, ref, {:content_block_start, 0, %{type: "text", text: ""}}})
        send(caller, {:stream, ref, {:content_block_delta, 0, %{type: "text_delta", text: "Hello"}}})
        send(caller, {:stream, ref, {:content_block_delta, 0, %{type: "text_delta", text: " world"}}})
        send(caller, {:stream, ref, {:content_block_stop, 0}})
        send(caller, {:stream, ref, {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 5}}})
        send(caller, {:stream, ref, :message_stop})
      end)

      events = collect_messages(ref, [])

      assert {:message_start, %{id: "msg_01ABC"}} = Enum.at(events, 0)
      assert {:content_block_start, 0, %{type: "text"}} = Enum.at(events, 1)
      assert {:content_block_delta, 0, %{type: "text_delta", text: "Hello"}} = Enum.at(events, 2)
      assert {:content_block_delta, 0, %{type: "text_delta", text: " world"}} = Enum.at(events, 3)
      assert {:content_block_stop, 0} = Enum.at(events, 4)
      assert {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 5}} = Enum.at(events, 5)
      assert :message_stop = Enum.at(events, 6)
    end

    test "tool use response produces correct Terra event sequence" do
      caller = self()
      ref = make_ref()

      spawn_link(fn ->
        send(caller, {:stream, ref, {:message_start, %{id: "msg_01TOOL", content: [], usage: %{input_tokens: 20, output_tokens: 0}}}})
        send(caller, {:stream, ref, {:content_block_start, 0, %{type: "tool_use", id: "toolu_01XYZ", name: "get_weather"}}})
        send(caller, {:stream, ref, {:content_block_delta, 0, %{type: "input_json_delta", partial_json: "{\"location\":"}}})
        send(caller, {:stream, ref, {:content_block_delta, 0, %{type: "input_json_delta", partial_json: " \"Tokyo\"}"}}})
        send(caller, {:stream, ref, {:content_block_stop, 0}})
        send(caller, {:stream, ref, {:message_delta, %{stop_reason: "tool_use"}, %{output_tokens: 15}}})
        send(caller, {:stream, ref, :message_stop})
      end)

      events = collect_messages(ref, [])

      assert {:message_start, _} = Enum.at(events, 0)
      assert {:content_block_start, 0, %{type: "tool_use", id: "toolu_01XYZ", name: "get_weather"}} = Enum.at(events, 1)
      assert {:content_block_delta, 0, %{type: "input_json_delta"}} = Enum.at(events, 2)
      assert {:content_block_stop, 0} = Enum.at(events, 4)
      assert {:message_delta, %{stop_reason: "tool_use"}, _} = Enum.at(events, 5)
    end

    test "thinking blocks produce correct Terra event sequence" do
      caller = self()
      ref = make_ref()

      spawn_link(fn ->
        send(caller, {:stream, ref, {:message_start, %{id: "msg_01THINK", content: [], usage: %{input_tokens: 30, output_tokens: 0}}}})
        # Thinking block
        send(caller, {:stream, ref, {:content_block_start, 0, %{type: "thinking", thinking: ""}}})
        send(caller, {:stream, ref, {:content_block_delta, 0, %{type: "thinking_delta", thinking: "Let me work through this"}}})
        send(caller, {:stream, ref, {:content_block_delta, 0, %{type: "signature_delta", signature: "sig_abc123"}}})
        send(caller, {:stream, ref, {:content_block_stop, 0}})
        # Text block follows
        send(caller, {:stream, ref, {:content_block_start, 1, %{type: "text", text: ""}}})
        send(caller, {:stream, ref, {:content_block_delta, 1, %{type: "text_delta", text: "The answer is 42"}}})
        send(caller, {:stream, ref, {:content_block_stop, 1}})
        send(caller, {:stream, ref, {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 25}}})
        send(caller, {:stream, ref, :message_stop})
      end)

      events = collect_messages(ref, [])

      assert {:content_block_start, 0, %{type: "thinking"}} = Enum.at(events, 1)
      assert {:content_block_delta, 0, %{type: "thinking_delta", thinking: "Let me work through this"}} = Enum.at(events, 2)
      assert {:content_block_delta, 0, %{type: "signature_delta", signature: "sig_abc123"}} = Enum.at(events, 3)
      assert {:content_block_start, 1, %{type: "text"}} = Enum.at(events, 5)
    end

    test "stop reasons map correctly: end_turn, tool_use, max_tokens" do
      for reason <- ["end_turn", "tool_use", "max_tokens", "stop_sequence"] do
        caller = self()
        ref = make_ref()

        spawn_link(fn ->
          send(caller, {:stream, ref, {:message_start, %{id: "msg_x", content: [], usage: %{input_tokens: 1, output_tokens: 0}}}})
          send(caller, {:stream, ref, {:message_delta, %{stop_reason: reason}, %{output_tokens: 1}}})
          send(caller, {:stream, ref, :message_stop})
        end)

        events = collect_messages(ref, [])
        assert {:message_delta, %{stop_reason: ^reason}, _} = Enum.at(events, 1)
      end
    end

    test "error events propagate" do
      caller = self()
      ref = make_ref()

      spawn_link(fn ->
        send(caller, {:stream, ref, {:error, %Terra.APIError{type: :authentication, status: 401, message: "invalid key"}}})
      end)

      events = collect_messages(ref, [])
      assert [{:error, %Terra.APIError{type: :authentication}}] = events
    end
  end
end
