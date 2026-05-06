defmodule Terra.Provider.GoogleTest do
  use ExUnit.Case, async: true

  alias Terra.Provider.Google
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
      assert Code.ensure_loaded?(Google)
      assert function_exported?(Google, :stream, 2)
      assert function_exported?(Google, :cancel, 1)
    end
  end

  describe "Terra event protocol conformance" do
    test "text response produces correct Terra event sequence" do
      caller = self()
      ref = make_ref()

      # Simulate what the Google provider would emit after translating
      # Gemini's parts-based streamGenerateContent format
      spawn_link(fn ->
        send(caller, {:stream, ref, {:message_start, %{id: "gemini-1", content: [], usage: %{input_tokens: 0, output_tokens: 0}}}})
        send(caller, {:stream, ref, {:content_block_start, 0, %{type: "text", text: ""}}})
        send(caller, {:stream, ref, {:content_block_delta, 0, %{type: "text_delta", text: "Hello"}}})
        send(caller, {:stream, ref, {:content_block_delta, 0, %{type: "text_delta", text: " from Gemini"}}})
        send(caller, {:stream, ref, {:content_block_stop, 0}})
        send(caller, {:stream, ref, {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 8}}})
        send(caller, {:stream, ref, :message_stop})
      end)

      events = collect_messages(ref, [])

      assert {:message_start, %{id: "gemini-1"}} = Enum.at(events, 0)
      assert {:content_block_start, 0, %{type: "text"}} = Enum.at(events, 1)
      assert {:content_block_delta, 0, %{type: "text_delta", text: "Hello"}} = Enum.at(events, 2)
      assert {:content_block_delta, 0, %{type: "text_delta", text: " from Gemini"}} = Enum.at(events, 3)
      assert {:content_block_stop, 0} = Enum.at(events, 4)
      assert {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 8}} = Enum.at(events, 5)
      assert :message_stop = Enum.at(events, 6)
    end

    test "function call response produces correct Terra event sequence" do
      caller = self()
      ref = make_ref()

      # Gemini emits functionCall parts which the provider maps to tool_use blocks.
      # Unlike Anthropic, Gemini delivers the full function args at once rather than
      # streaming deltas — the provider emits a single input_json_delta with the full payload.
      spawn_link(fn ->
        send(caller, {:stream, ref, {:message_start, %{id: "gemini-2", content: [], usage: %{input_tokens: 0, output_tokens: 0}}}})
        send(caller, {:stream, ref, {:content_block_start, 0, %{type: "tool_use", id: "call_1", name: "get_weather"}}})
        send(caller, {:stream, ref, {:content_block_delta, 0, %{type: "input_json_delta", partial_json: "{\"location\":\"Paris\"}"}}})
        send(caller, {:stream, ref, {:content_block_stop, 0}})
        send(caller, {:stream, ref, {:message_delta, %{stop_reason: "tool_use"}, %{output_tokens: 12}}})
        send(caller, {:stream, ref, :message_stop})
      end)

      events = collect_messages(ref, [])

      assert {:content_block_start, 0, %{type: "tool_use", id: "call_1", name: "get_weather"}} = Enum.at(events, 1)
      assert {:content_block_delta, 0, %{type: "input_json_delta", partial_json: "{\"location\":\"Paris\"}"}} = Enum.at(events, 2)
      assert {:content_block_stop, 0} = Enum.at(events, 3)
      assert {:message_delta, %{stop_reason: "tool_use"}, _} = Enum.at(events, 4)
    end

    test "thinking blocks produce correct Terra event sequence" do
      caller = self()
      ref = make_ref()

      # Gemini's thought parts get translated into Terra's thinking blocks
      spawn_link(fn ->
        send(caller, {:stream, ref, {:message_start, %{id: "gemini-3", content: [], usage: %{input_tokens: 0, output_tokens: 0}}}})
        send(caller, {:stream, ref, {:content_block_start, 0, %{type: "thinking", thinking: ""}}})
        send(caller, {:stream, ref, {:content_block_delta, 0, %{type: "thinking_delta", thinking: "Reasoning..."}}})
        send(caller, {:stream, ref, {:content_block_stop, 0}})
        send(caller, {:stream, ref, {:content_block_start, 1, %{type: "text", text: ""}}})
        send(caller, {:stream, ref, {:content_block_delta, 1, %{type: "text_delta", text: "Final answer"}}})
        send(caller, {:stream, ref, {:content_block_stop, 1}})
        send(caller, {:stream, ref, {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 20}}})
        send(caller, {:stream, ref, :message_stop})
      end)

      events = collect_messages(ref, [])

      assert {:content_block_start, 0, %{type: "thinking"}} = Enum.at(events, 1)
      assert {:content_block_delta, 0, %{type: "thinking_delta", thinking: "Reasoning..."}} = Enum.at(events, 2)
      assert {:content_block_start, 1, %{type: "text"}} = Enum.at(events, 4)
    end

    test "Gemini finishReason maps to Terra stop_reason" do
      # Gemini uses STOP / MAX_TOKENS / SAFETY; provider normalizes to Terra's vocabulary
      for reason <- ["end_turn", "tool_use", "max_tokens"] do
        caller = self()
        ref = make_ref()

        spawn_link(fn ->
          send(caller, {:stream, ref, {:message_start, %{id: "gemini-x", content: [], usage: %{input_tokens: 1, output_tokens: 0}}}})
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

  describe "format_messages/1 — Terra → Gemini translation" do
    test "user role stays user, assistant role becomes model" do
      msgs = [
        %{role: "user", content: "hi"},
        %{role: "assistant", content: "hello"}
      ]

      [u, a] = Google.format_messages(msgs)
      assert u.role == "user"
      assert a.role == "model"
    end

    test "tool_use becomes functionCall part" do
      msgs = [
        %{
          role: "assistant",
          content: [%{type: "tool_use", id: "t1", name: "get_weather", input: %{location: "Tokyo"}}]
        }
      ]

      [out] = Google.format_messages(msgs)
      assert [%{functionCall: fc}] = out.parts
      assert fc.name == "get_weather"
      assert fc.args == %{location: "Tokyo"}
    end

    test "tool_result resolves name from prior assistant tool_use via tool_use_id" do
      msgs = [
        %{
          role: "assistant",
          content: [%{type: "tool_use", id: "t1", name: "get_weather", input: %{}}]
        },
        %{
          role: "user",
          content: [%{type: "tool_result", tool_use_id: "t1", content: "72°F"}]
        }
      ]

      [_assistant, user] = Google.format_messages(msgs)
      assert [%{functionResponse: fr}] = user.parts
      assert fr.name == "get_weather"
      assert fr.response == %{result: "72°F"}
    end

    test "tool_result without matching tool_use falls back to id as name" do
      msgs = [
        %{role: "user", content: [%{type: "tool_result", tool_use_id: "orphan", content: "x"}]}
      ]

      [out] = Google.format_messages(msgs)
      assert [%{functionResponse: fr}] = out.parts
      assert fr.name == "orphan"
    end

    test "Terra.Document is rendered as a text part" do
      doc = Document.new("forecast", "ctx", "72°F sunny")

      msgs = [%{role: "user", content: [doc]}]

      [out] = Google.format_messages(msgs)
      assert [%{text: text}] = out.parts
      assert text =~ ~s|<document title="forecast"|
      assert text =~ "72°F sunny"
    end

    test "thinking blocks are dropped" do
      msgs = [
        %{
          role: "assistant",
          content: [
            %{type: "thinking", thinking: "internal"},
            %{type: "text", text: "answer"}
          ]
        }
      ]

      [out] = Google.format_messages(msgs)
      # thinking dropped, only text remains
      assert out.parts == [%{text: "answer"}]
    end

    test "messages with only thinking content are dropped entirely" do
      msgs = [
        %{role: "assistant", content: [%{type: "thinking", thinking: "internal"}]}
      ]

      assert Google.format_messages(msgs) == []
    end

    test "tool_response wraps strings under :result for object compliance" do
      msgs = [
        %{role: "user", content: [%{type: "tool_result", tool_use_id: "x", content: "plain"}]}
      ]

      [out] = Google.format_messages(msgs)
      assert [%{functionResponse: %{response: %{result: "plain"}}}] = out.parts
    end

    test "tool_response passes through map content directly" do
      msgs = [
        %{role: "user", content: [%{type: "tool_result", tool_use_id: "x", content: %{temp: 72}}]}
      ]

      [out] = Google.format_messages(msgs)
      assert [%{functionResponse: %{response: %{temp: 72}}}] = out.parts
    end
  end
end
