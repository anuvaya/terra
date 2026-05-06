defmodule Terra.Provider.AnthropicLiveTest do
  use ExUnit.Case

  @moduletag :live

  @api_key System.get_env("ANTHROPIC_API_KEY", "")

  # ── Text Streaming ─────────────────────────────────────

  describe "text streaming" do
    test "simple text response" do
      events = stream!(%{
        model: "claude-haiku-4-5-20251001",
        max_tokens: 100,
        system: "Reply in exactly one short sentence.",
        messages: [%{"role" => "user", "content" => "What is 2+2?"}]
      })

      assert_event_sequence(events, [
        :message_start, :content_block_start, :content_block_delta,
        :content_block_stop, :message_delta, :message_stop
      ])

      text = extract_text(events)
      assert String.length(text) > 0
      IO.puts("  Text: #{text}")

      # Usage tracking
      {:message_start, msg} = find_event(events, :message_start)
      assert msg.usage.input_tokens > 0

      {:message_delta, delta, usage} = find_event(events, :message_delta)
      assert delta.stop_reason == "end_turn"
      assert usage.output_tokens > 0
    end

    test "multi-turn conversation" do
      events = stream!(%{
        model: "claude-haiku-4-5-20251001",
        max_tokens: 100,
        messages: [
          %{"role" => "user", "content" => "My name is Alice."},
          %{"role" => "assistant", "content" => "Hello Alice!"},
          %{"role" => "user", "content" => "What's my name? Reply with just the name."}
        ]
      })

      text = extract_text(events)
      assert text =~ "Alice"
      IO.puts("  Text: #{text}")
    end

    test "long response produces many deltas" do
      events = stream!(%{
        model: "claude-haiku-4-5-20251001",
        max_tokens: 500,
        messages: [%{"role" => "user", "content" => "Write a haiku about Elixir programming."}]
      })

      deltas = Enum.filter(events, &match?({:content_block_delta, _, %{type: "text_delta"}}, &1))
      assert length(deltas) > 1
      IO.puts("  #{length(deltas)} text deltas received")

      text = extract_text(events)
      IO.puts("  Text: #{text}")
    end
  end

  # ── Tool Use ───────────────────────────────────────────

  describe "tool use" do
    test "single tool call with JSON input" do
      events = stream!(%{
        model: "claude-haiku-4-5-20251001",
        max_tokens: 200,
        system: "Use the get_weather tool to answer.",
        messages: [%{"role" => "user", "content" => "What's the weather in Tokyo?"}],
        tools: [weather_tool()],
        tool_choice: %{type: "tool", name: "get_weather"}
      })

      # Tool use block
      {:content_block_start, _, block} = find_event(events, :tool_use_start)
      assert block.name == "get_weather"
      assert is_binary(block.id) and block.id != ""
      IO.puts("  Tool: #{block.name}, ID: #{block.id}")

      # Accumulated JSON input
      json = extract_tool_input(events)
      {:ok, parsed} = Jason.decode(json)
      assert Map.has_key?(parsed, "location")
      IO.puts("  Input: #{json}")

      # Stop reason
      {:message_delta, delta, _} = find_event(events, :message_delta)
      assert delta.stop_reason == "tool_use"
    end

    test "tool with no required arguments" do
      events = stream!(%{
        model: "claude-haiku-4-5-20251001",
        max_tokens: 200,
        system: "Use the list_items tool.",
        messages: [%{"role" => "user", "content" => "Show me all items"}],
        tools: [%{
          name: "list_items",
          description: "List all available items. Takes no arguments.",
          input_schema: %{type: "object", properties: %{}}
        }],
        tool_choice: %{type: "tool", name: "list_items"}
      })

      {:content_block_start, _, block} = find_event(events, :tool_use_start)
      assert block.name == "list_items"

      {:message_delta, delta, _} = find_event(events, :message_delta)
      assert delta.stop_reason == "tool_use"
      IO.puts("  Tool called: #{block.name}")
    end

    test "multiple tool calls" do
      events = stream!(%{
        model: "claude-haiku-4-5-20251001",
        max_tokens: 400,
        system: "Use both tools to answer. Call get_weather for each city.",
        messages: [%{"role" => "user", "content" => "Weather in Tokyo and London?"}],
        tools: [weather_tool()]
      })

      tool_starts = Enum.filter(events, fn
        {:content_block_start, _, %{type: "tool_use"}} -> true
        _ -> false
      end)

      IO.puts("  Tool calls: #{length(tool_starts)}")
      for {:content_block_start, idx, block} <- tool_starts do
        IO.puts("    [#{idx}] #{block.name} (#{block.id})")
      end

      # May get 1 or 2 tool calls depending on model behavior
      assert length(tool_starts) >= 1
    end

    test "tool call with complex nested input" do
      events = stream!(%{
        model: "claude-haiku-4-5-20251001",
        max_tokens: 300,
        system: "Use the create_event tool.",
        messages: [%{"role" => "user", "content" => "Create a meeting tomorrow at 3pm with Alice and Bob about project planning"}],
        tools: [%{
          name: "create_event",
          description: "Create a calendar event",
          input_schema: %{
            type: "object",
            properties: %{
              title: %{type: "string"},
              time: %{type: "string"},
              attendees: %{type: "array", items: %{type: "string"}}
            },
            required: ["title", "time", "attendees"]
          }
        }],
        tool_choice: %{type: "tool", name: "create_event"}
      })

      json = extract_tool_input(events)
      {:ok, parsed} = Jason.decode(json)
      assert is_binary(parsed["title"])
      assert is_list(parsed["attendees"])
      IO.puts("  Input: #{json}")
    end
  end

  # ── Extended Thinking ──────────────────────────────────

  describe "extended thinking" do
    test "thinking blocks with text output" do
      events = stream!(%{
        model: "claude-sonnet-4-5-20250929",
        max_tokens: 1024,
        thinking: %{type: "enabled", budget_tokens: 1024},
        messages: [%{"role" => "user", "content" => "What is 15 * 37? Think step by step."}],
        config: %{api_key: @api_key, beta: ["interleaved-thinking-2025-05-14"]}
      })

      event_types = classify_events(events)
      IO.puts("  Events: #{inspect(event_types)}")

      # Should have thinking block(s)
      thinking_starts = Enum.filter(events, fn
        {:content_block_start, _, %{type: "thinking"}} -> true
        _ -> false
      end)
      assert length(thinking_starts) > 0
      IO.puts("  Thinking blocks: #{length(thinking_starts)}")

      # Should have thinking deltas
      thinking_text =
        events
        |> Enum.filter(fn
          {:content_block_delta, _, %{type: "thinking_delta"}} -> true
          _ -> false
        end)
        |> Enum.map(fn {:content_block_delta, _, %{thinking: t}} -> t end)
        |> Enum.join()

      assert String.length(thinking_text) > 0
      IO.puts("  Thinking: #{String.slice(thinking_text, 0, 100)}...")

      # Should have signature
      signatures = Enum.filter(events, fn
        {:content_block_delta, _, %{type: "signature_delta"}} -> true
        _ -> false
      end)
      assert length(signatures) > 0
      IO.puts("  Signatures: #{length(signatures)}")

      # Should also have text output
      text = extract_text(events)
      assert String.length(text) > 0
      assert text =~ "555"
      IO.puts("  Answer: #{text}")
    end
  end

  # ── Error Handling ─────────────────────────────────────

  describe "error handling" do
    test "invalid API key returns authentication error" do
      events = stream!(%{
        model: "claude-haiku-4-5-20251001",
        max_tokens: 10,
        messages: [%{"role" => "user", "content" => "hi"}],
        config: %{api_key: "sk-ant-invalid-key"}
      })

      assert [{:error, error}] = events
      assert %Terra.APIError{} = error
      assert error.type == :authentication
      assert error.status == 401
      IO.puts("  Error: #{error.message}")
    end

    test "invalid model returns error" do
      events = stream!(%{
        model: "nonexistent-model-xyz",
        max_tokens: 10,
        messages: [%{"role" => "user", "content" => "hi"}]
      })

      assert [{:error, error}] = events
      assert %Terra.APIError{} = error
      IO.puts("  Error: #{error.type} - #{error.message}")
    end

    test "empty messages returns error" do
      events = stream!(%{
        model: "claude-haiku-4-5-20251001",
        max_tokens: 10,
        messages: []
      })

      assert [{:error, error}] = events
      assert %Terra.APIError{} = error
      IO.puts("  Error: #{error.type} - #{error.message}")
    end
  end

  # ── Event Ordering & Correctness ───────────────────────

  describe "event protocol correctness" do
    test "events arrive in correct order" do
      events = stream!(%{
        model: "claude-haiku-4-5-20251001",
        max_tokens: 50,
        messages: [%{"role" => "user", "content" => "Say hello"}]
      })

      types = classify_events(events)

      # message_start is always first
      assert hd(types) == :message_start

      # message_stop is always last
      assert List.last(types) == :message_stop

      # message_delta comes right before message_stop
      {before_stop, _} = Enum.split(types, -1)
      assert List.last(before_stop) == :message_delta

      # content_block_start always precedes its deltas
      first_start_idx = Enum.find_index(types, &(&1 == :content_block_start))
      first_delta_idx = Enum.find_index(types, &(&1 == :content_block_delta))
      assert first_start_idx < first_delta_idx
    end

    test "content block indices are consistent" do
      events = stream!(%{
        model: "claude-haiku-4-5-20251001",
        max_tokens: 100,
        messages: [%{"role" => "user", "content" => "Say hi"}]
      })

      # All text deltas should reference the same block index as the start
      {:content_block_start, start_idx, _} =
        Enum.find(events, &match?({:content_block_start, _, %{type: "text"}}, &1))

      text_deltas = Enum.filter(events, &match?({:content_block_delta, _, %{type: "text_delta"}}, &1))

      for {:content_block_delta, idx, _} <- text_deltas do
        assert idx == start_idx
      end

      # Stop should also match
      {:content_block_stop, stop_idx} =
        Enum.find(events, &match?({:content_block_stop, ^start_idx}, &1))

      assert stop_idx == start_idx
    end

    test "message_start contains valid metadata" do
      events = stream!(%{
        model: "claude-haiku-4-5-20251001",
        max_tokens: 10,
        messages: [%{"role" => "user", "content" => "hi"}]
      })

      {:message_start, msg} = find_event(events, :message_start)

      assert is_binary(msg.id)
      assert msg.id =~ "msg_"
      assert is_map(msg.usage)
      assert is_integer(msg.usage.input_tokens)
      IO.puts("  Message ID: #{msg.id}")
    end
  end

  # ── Helpers ────────────────────────────────────────────

  defp stream!(params) do
    caller = self()

    params =
      params
      |> Map.put_new(:tools, [])
      |> Map.put_new(:config, %{api_key: @api_key})

    # Ensure config has api_key
    config = Map.get(params, :config, %{})
    params = %{params | config: Map.put_new(config, :api_key, @api_key)}

    {:ok, ref} = Terra.Provider.Anthropic.stream(caller, params)
    collect_events(ref, [], 30_000)
  end

  defp collect_events(ref, acc, timeout) do
    receive do
      {:stream, ^ref, event} ->
        case event do
          :message_stop -> acc ++ [event]
          {:error, _} -> acc ++ [event]
          _ -> collect_events(ref, acc ++ [event], timeout)
        end
    after
      timeout ->
        IO.puts("  TIMEOUT waiting for events!")
        acc
    end
  end

  defp extract_text(events) do
    events
    |> Enum.filter(&match?({:content_block_delta, _, %{type: "text_delta"}}, &1))
    |> Enum.map(fn {:content_block_delta, _, %{text: t}} -> t end)
    |> Enum.join()
  end

  defp extract_tool_input(events) do
    events
    |> Enum.filter(&match?({:content_block_delta, _, %{type: "input_json_delta"}}, &1))
    |> Enum.map(fn {:content_block_delta, _, %{partial_json: j}} -> j end)
    |> Enum.join()
  end

  defp classify_events(events) do
    Enum.map(events, fn
      {:message_start, _} -> :message_start
      {:content_block_start, _, %{type: "tool_use"}} -> :tool_use_start
      {:content_block_start, _, %{type: "thinking"}} -> :thinking_start
      {:content_block_start, _, _} -> :content_block_start
      {:content_block_delta, _, %{type: "text_delta"}} -> :content_block_delta
      {:content_block_delta, _, %{type: "input_json_delta"}} -> :input_json_delta
      {:content_block_delta, _, %{type: "thinking_delta"}} -> :thinking_delta
      {:content_block_delta, _, %{type: "signature_delta"}} -> :signature_delta
      {:content_block_stop, _} -> :content_block_stop
      {:message_delta, _, _} -> :message_delta
      :message_stop -> :message_stop
      :ping -> :ping
      {:error, _} -> :error
      _ -> :unknown
    end)
  end

  defp find_event(events, :message_start) do
    Enum.find(events, &match?({:message_start, _}, &1))
  end

  defp find_event(events, :message_delta) do
    Enum.find(events, &match?({:message_delta, _, _}, &1))
  end

  defp find_event(events, :tool_use_start) do
    Enum.find(events, &match?({:content_block_start, _, %{type: "tool_use"}}, &1))
  end

  defp assert_event_sequence(events, required_types) do
    types = classify_events(events)

    for type <- required_types do
      assert type in types,
        "Expected #{inspect(type)} in event stream, got: #{inspect(types)}"
    end
  end

  defp weather_tool do
    %{
      name: "get_weather",
      description: "Get current weather for a location",
      input_schema: %{
        type: "object",
        properties: %{
          location: %{type: "string", description: "City name"}
        },
        required: ["location"]
      }
    }
  end
end
