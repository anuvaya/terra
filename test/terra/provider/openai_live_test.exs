defmodule Terra.Provider.OpenAILiveTest do
  use ExUnit.Case

  @moduletag :live

  @api_key System.get_env("OPENAI_API_KEY", "")

  # ── Text Streaming ─────────────────────────────────────

  describe "text streaming" do
    test "simple text response" do
      events = stream!(%{
        model: "gpt-4o-mini",
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

      {:message_delta, delta, usage} = find_event(events, :message_delta)
      assert delta.stop_reason == "end_turn"
      assert usage.output_tokens > 0
    end

    test "multi-turn conversation" do
      events = stream!(%{
        model: "gpt-4o-mini",
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
        model: "gpt-4o-mini",
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
        model: "gpt-4o-mini",
        max_tokens: 200,
        system: "Use the get_weather tool to answer.",
        messages: [%{"role" => "user", "content" => "What's the weather in Tokyo?"}],
        tools: [weather_tool()],
        tool_choice: %{type: "function", function: %{name: "get_weather"}}
      })

      {:content_block_start, _, block} = find_event(events, :tool_use_start)
      assert block.name == "get_weather"
      assert is_binary(block.id) and block.id != ""
      IO.puts("  Tool: #{block.name}, ID: #{block.id}")

      json = extract_tool_input(events)
      {:ok, parsed} = Jason.decode(json)
      assert Map.has_key?(parsed, "location")
      IO.puts("  Input: #{json}")

      {:message_delta, delta, _} = find_event(events, :message_delta)
      assert delta.stop_reason == "tool_use"
    end

    test "tool with no required arguments" do
      events = stream!(%{
        model: "gpt-4o-mini",
        max_tokens: 200,
        system: "Use the list_items tool.",
        messages: [%{"role" => "user", "content" => "Show me all items"}],
        tools: [%{
          name: "list_items",
          description: "List all available items. Takes no arguments.",
          input_schema: %{type: "object", properties: %{}}
        }],
        tool_choice: %{type: "function", function: %{name: "list_items"}}
      })

      {:content_block_start, _, block} = find_event(events, :tool_use_start)
      assert block.name == "list_items"

      {:message_delta, delta, _} = find_event(events, :message_delta)
      assert delta.stop_reason == "tool_use"
      IO.puts("  Tool called: #{block.name}")
    end

    test "multiple tool calls" do
      events = stream!(%{
        model: "gpt-4o-mini",
        max_tokens: 400,
        system: "Use the get_weather tool for each city the user mentions.",
        messages: [%{"role" => "user", "content" => "Weather in Tokyo and London?"}],
        tools: [weather_tool()],
        parallel_tool_calls: true
      })

      tool_starts = Enum.filter(events, fn
        {:content_block_start, _, %{type: "tool_use"}} -> true
        _ -> false
      end)

      IO.puts("  Tool calls: #{length(tool_starts)}")
      for {:content_block_start, idx, block} <- tool_starts do
        IO.puts("    [#{idx}] #{block.name} (#{block.id})")
      end

      assert length(tool_starts) >= 1
    end

    test "tool call with complex nested input" do
      events = stream!(%{
        model: "gpt-4o-mini",
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
        tool_choice: %{type: "function", function: %{name: "create_event"}}
      })

      json = extract_tool_input(events)
      {:ok, parsed} = Jason.decode(json)
      assert is_binary(parsed["title"])
      assert is_list(parsed["attendees"])
      IO.puts("  Input: #{json}")
    end

    test "tool call args stream as multiple JSON deltas" do
      events = stream!(%{
        model: "gpt-4o-mini",
        max_tokens: 300,
        system: "Use the create_event tool.",
        messages: [%{"role" => "user", "content" => "Create a meeting tomorrow at 3pm with Alice, Bob, Carol, Dave, and Eve about quarterly planning"}],
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
        tool_choice: %{type: "function", function: %{name: "create_event"}}
      })

      json_deltas = Enum.filter(events, &match?({:content_block_delta, _, %{type: "input_json_delta"}}, &1))
      IO.puts("  JSON deltas: #{length(json_deltas)}")
      assert length(json_deltas) >= 1

      json = extract_tool_input(events)
      {:ok, _parsed} = Jason.decode(json)
    end
  end

  # ── Error Handling ─────────────────────────────────────

  describe "error handling" do
    test "invalid API key returns authentication error" do
      events = stream!(%{
        model: "gpt-4o-mini",
        max_tokens: 10,
        messages: [%{"role" => "user", "content" => "hi"}],
        config: %{api_key: "sk-invalid-key-xyz"}
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
        model: "gpt-4o-mini",
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
        model: "gpt-4o-mini",
        max_tokens: 50,
        messages: [%{"role" => "user", "content" => "Say hello"}]
      })

      types = classify_events(events)

      assert hd(types) == :message_start
      assert List.last(types) == :message_stop

      {before_stop, _} = Enum.split(types, -1)
      assert List.last(before_stop) == :message_delta

      first_start_idx = Enum.find_index(types, &(&1 == :content_block_start))
      first_delta_idx = Enum.find_index(types, &(&1 == :content_block_delta))
      assert first_start_idx < first_delta_idx
    end

    test "content block indices are consistent" do
      events = stream!(%{
        model: "gpt-4o-mini",
        max_tokens: 100,
        messages: [%{"role" => "user", "content" => "Say hi"}]
      })

      {:content_block_start, start_idx, _} =
        Enum.find(events, &match?({:content_block_start, _, %{type: "text"}}, &1))

      text_deltas = Enum.filter(events, &match?({:content_block_delta, _, %{type: "text_delta"}}, &1))

      for {:content_block_delta, idx, _} <- text_deltas do
        assert idx == start_idx
      end

      {:content_block_stop, stop_idx} =
        Enum.find(events, &match?({:content_block_stop, ^start_idx}, &1))

      assert stop_idx == start_idx
    end

    test "message_start contains valid metadata" do
      events = stream!(%{
        model: "gpt-4o-mini",
        max_tokens: 10,
        messages: [%{"role" => "user", "content" => "hi"}]
      })

      {:message_start, msg} = find_event(events, :message_start)

      assert is_binary(msg.id)
      assert is_map(msg.usage)
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

    config = Map.get(params, :config, %{})
    params = %{params | config: Map.put_new(config, :api_key, @api_key)}

    {:ok, ref} = Terra.Provider.OpenAI.stream(caller, params)
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
      {:content_block_start, _, _} -> :content_block_start
      {:content_block_delta, _, %{type: "text_delta"}} -> :content_block_delta
      {:content_block_delta, _, %{type: "input_json_delta"}} -> :input_json_delta
      {:content_block_stop, _} -> :content_block_stop
      {:message_delta, _, _} -> :message_delta
      :message_stop -> :message_stop
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
