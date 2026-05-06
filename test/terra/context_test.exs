defmodule Terra.ContextTest do
  use ExUnit.Case, async: true

  alias Terra.Context
  alias Terra.Document

  # ── Phase A: Builder ──────────────────────────────────────

  describe "builder" do
    test "new/0 creates empty context" do
      ctx = Context.new()

      assert ctx.system == nil
      assert ctx.documents == []
      assert ctx.history == []
      assert ctx.messages == []
      assert ctx.model == %{}
    end

    test "system/2 sets the system prompt" do
      ctx =
        Context.new()
        |> Context.system("You are a weather forecaster")

      assert ctx.system == "You are a weather forecaster"
    end

    test "document/2 adds a Terra.Document" do
      doc = Document.new("forecast", "Weather forecast", "forecast data", cache: :ephemeral)

      ctx =
        Context.new()
        |> Context.document(doc)

      assert [^doc] = ctx.documents
    end

    test "multiple documents preserve order" do
      d1 = Document.new("a", "ctx", "data a")
      d2 = Document.new("b", "ctx", "data b")

      ctx =
        Context.new()
        |> Context.document(d1)
        |> Context.document(d2)

      assert [^d1, ^d2] = ctx.documents
    end

    test "history/2 sets the interactions list" do
      messages = [
        %{role: "user", content: "hello"},
        %{role: "assistant", content: "hi there"}
      ]

      ctx =
        Context.new()
        |> Context.history(messages)

      assert ctx.history == messages
    end

    test "model/2 sets model configuration" do
      ctx =
        Context.new()
        |> Context.model(%{
          model: "claude-sonnet-4-5-20250929",
          max_tokens: 2048,
          thinking: %{type: "enabled", budget_tokens: 1024}
        })

      assert ctx.model.model == "claude-sonnet-4-5-20250929"
      assert ctx.model.max_tokens == 2048
    end

    test "full pipeline builds complete context" do
      doc = Document.new("forecast", "Forecast", "data")

      ctx =
        Context.new()
        |> Context.system("You are helpful")
        |> Context.document(doc)
        |> Context.history([%{role: "user", content: "hi"}])
        |> Context.model(%{model: "mock", max_tokens: 100})
        |> Context.build()

      assert ctx.system == "You are helpful"
      assert length(ctx.documents) == 1
      assert length(ctx.messages) == 1
      assert ctx.model.model == "mock"
    end
  end

  # ── Phase A2: nil-safe document ───────────────────────────

  describe "nil-safe document" do
    test "document/2 with nil is a no-op" do
      ctx =
        Context.new()
        |> Context.document(nil)

      assert ctx.documents == []
    end

    test "nil documents skipped in pipeline" do
      doc = Document.new("forecast", "Forecast", "data")

      ctx =
        Context.new()
        |> Context.document(nil)
        |> Context.document(doc)
        |> Context.document(nil)

      assert [^doc] = ctx.documents
    end
  end

  # ── Phase B: build/1,2 ──────────────────────────────────

  describe "build/1,2" do
    test "documents are injected into first user message" do
      doc = Document.new("forecast", "Weather forecast", "forecast data", cache: :ephemeral)

      ctx =
        Context.new()
        |> Context.document(doc)
        |> Context.history([%{role: "user", content: "hello"}])
        |> Context.build()

      [first | _] = ctx.messages

      assert injected_documents(first.content) == [doc]
      assert %{type: "text", text: "hello"} in first.content
    end

    test "no history — creates synthetic user message with documents" do
      doc = Document.new("forecast", "Forecast", "data")

      ctx =
        Context.new()
        |> Context.document(doc)
        |> Context.build()

      assert [%{role: "user", content: content}] = ctx.messages
      assert injected_documents(content) == [doc]
    end

    test "history with list content preserved as-is" do
      ctx =
        Context.new()
        |> Context.history([
          %{role: "user", content: [%{type: "text", text: "hello"}]},
          %{role: "assistant", content: [%{type: "text", text: "hi"}]}
        ])
        |> Context.build()

      assert [
               %{role: "user", content: [%{type: "text", text: "hello"}]},
               %{role: "assistant", content: [%{type: "text", text: "hi"}]}
             ] = ctx.messages
    end

    test "empty context returns empty messages" do
      ctx = Context.new() |> Context.build()
      assert ctx.messages == []
    end

    test "build returns a Context struct" do
      ctx =
        Context.new()
        |> Context.system("test")
        |> Context.history([%{role: "user", content: "hi"}])
        |> Context.build()

      assert %Context{} = ctx
      assert ctx.system == "test"
      assert length(ctx.messages) == 1
    end
  end

  # ── Phase C: Aging ────────────────────────────────────────

  describe "aging" do
    defmodule AgingRegistry do
      use Terra.ToolRegistry

      @impl true
      def tools(_state) do
        [
          %{
            name: "get_weather",
            description: "Get weather",
            input_schema: %{},
            expiry_distance: 2,
            pruning_distance: 4,
            result_template: "Weather: <%= @result %>",
            expiry_message: "[Weather expired. Input: <%= inspect(@input) %>]"
          },
          %{
            name: "fetch_forecast",
            description: "Fetch chart",
            input_schema: %{},
            expiry_distance: 3,
            pruning_distance: :infinity
          },
          %{
            name: "simple_tool",
            description: "No aging",
            input_schema: %{}
          },
          %{
            name: "hinted_tool",
            description: "Tool with tail hints",
            input_schema: %{},
            tail_hints: "Focus on the key findings."
          }
        ]
      end

      @impl true
      def execute(_, _, state), do: {:ok, "result", state}
    end

    defp make_state(turn_count) do
      %Terra.Agent.State{
        turn_count: turn_count,
        registries: [AgingRegistry]
      }
    end

    # Distance is measured by counting assistant messages from the end.
    # Last assistant = distance 0, previous assistant = distance 1, etc.

    test "active tool result passes through unchanged" do
      # Content is already rendered by ToolRegistry.run/4 at execute time
      history = [
        %{role: "user", content: "what's the weather?"},
        %{role: "assistant", content: [
          %{type: "tool_use", id: "t1", name: "get_weather", input: %{"location" => "SF"}}
        ]},
        %{role: "user", content: [
          %{type: "tool_result", tool_use_id: "t1", content: "Weather: 72°F and sunny"}
        ]},
        %{role: "assistant", content: [%{type: "text", text: "It's nice in SF!"}]}
      ]

      # Distance 0 from last assistant — tool is at distance 1
      # expiry_distance is 2, so distance 1 < 2 → active → pass through
      ctx = Context.new() |> Context.history(history) |> Context.age_tools(make_state(2)) |> Context.build()

      tool_result_msg = Enum.at(ctx.messages, 2)
      [result_block] = tool_result_msg.content

      assert result_block.content == "Weather: 72°F and sunny"
    end

    test "expired tool result rendered with expiry_message" do
      history = [
        %{role: "user", content: "weather?"},
        %{role: "assistant", content: [
          %{type: "tool_use", id: "t1", name: "get_weather", input: %{"location" => "SF"}}
        ]},
        %{role: "user", content: [
          %{type: "tool_result", tool_use_id: "t1", content: "72°F"}
        ]},
        %{role: "assistant", content: [%{type: "text", text: "Nice!"}]},
        # 2 more turns to push distance to 2+
        %{role: "user", content: "how about now?"},
        %{role: "assistant", content: [%{type: "text", text: "Still nice"}]},
        %{role: "user", content: "and now?"},
        %{role: "assistant", content: [%{type: "text", text: "Yep"}]}
      ]

      # Tool is at assistant index 0 (distance 3 from end)
      # expiry_distance 2 ≤ 3 < pruning_distance 4 → expired
      ctx = Context.new() |> Context.history(history) |> Context.age_tools(make_state(4)) |> Context.build()

      tool_result_msg = Enum.at(ctx.messages, 2)
      [result_block] = tool_result_msg.content

      assert result_block.content =~ "Weather expired"
      assert result_block.content =~ "SF"
    end

    test "pruned tool result omitted entirely" do
      history = [
        %{role: "user", content: "weather?"},
        %{role: "assistant", content: [
          %{type: "tool_use", id: "t1", name: "get_weather", input: %{"location" => "SF"}}
        ]},
        %{role: "user", content: [
          %{type: "tool_result", tool_use_id: "t1", content: "72°F"}
        ]},
        %{role: "assistant", content: [%{type: "text", text: "Nice!"}]},
        %{role: "user", content: "a"},
        %{role: "assistant", content: [%{type: "text", text: "b"}]},
        %{role: "user", content: "c"},
        %{role: "assistant", content: [%{type: "text", text: "d"}]},
        %{role: "user", content: "e"},
        %{role: "assistant", content: [%{type: "text", text: "f"}]}
      ]

      # Tool at assistant distance 4 — pruning_distance is 4 → pruned
      ctx = Context.new() |> Context.history(history) |> Context.age_tools(make_state(5)) |> Context.build()

      # The tool_use assistant message and tool_result user message should be gone
      refute Enum.any?(ctx.messages, fn msg ->
        is_list(msg.content) and
          Enum.any?(msg.content, fn
            %{type: "tool_use"} -> true
            _ -> false
          end)
      end)
    end

    test "tool with pruning_distance: :infinity never pruned" do
      history = [
        %{role: "user", content: "chart?"},
        %{role: "assistant", content: [
          %{type: "tool_use", id: "t1", name: "fetch_forecast", input: %{}}
        ]},
        %{role: "user", content: [
          %{type: "tool_result", tool_use_id: "t1", content: "chart data"}
        ]},
        %{role: "assistant", content: [%{type: "text", text: "Here's your chart"}]},
        # Many turns later...
        %{role: "user", content: "a"},
        %{role: "assistant", content: [%{type: "text", text: "b"}]},
        %{role: "user", content: "c"},
        %{role: "assistant", content: [%{type: "text", text: "d"}]},
        %{role: "user", content: "e"},
        %{role: "assistant", content: [%{type: "text", text: "f"}]},
        %{role: "user", content: "g"},
        %{role: "assistant", content: [%{type: "text", text: "h"}]}
      ]

      ctx = Context.new() |> Context.history(history) |> Context.age_tools(make_state(10)) |> Context.build()

      # Tool result should still exist (infinity pruning)
      assert Enum.any?(ctx.messages, fn msg ->
        is_list(msg.content) and
          Enum.any?(msg.content, fn
            %{type: "tool_result"} -> true
            _ -> false
          end)
      end)
    end

    test "tool without expiry_distance stays active forever" do
      history = [
        %{role: "user", content: "do it"},
        %{role: "assistant", content: [
          %{type: "tool_use", id: "t1", name: "simple_tool", input: %{}}
        ]},
        %{role: "user", content: [
          %{type: "tool_result", tool_use_id: "t1", content: "done"}
        ]},
        %{role: "assistant", content: [%{type: "text", text: "ok"}]},
        %{role: "user", content: "a"},
        %{role: "assistant", content: [%{type: "text", text: "b"}]},
        %{role: "user", content: "c"},
        %{role: "assistant", content: [%{type: "text", text: "d"}]}
      ]

      ctx = Context.new() |> Context.history(history) |> Context.age_tools(make_state(10)) |> Context.build()

      # Result should be unchanged — no aging applied
      tool_result_msg = Enum.find(ctx.messages, fn msg ->
        is_list(msg.content) and
          Enum.any?(msg.content, fn
            %{type: "tool_result"} -> true
            _ -> false
          end)
      end)

      assert tool_result_msg != nil
      [result] = tool_result_msg.content
      assert result.content == "done"
    end

    test "no state passed — no aging applied" do
      history = [
        %{role: "user", content: "hi"},
        %{role: "assistant", content: [%{type: "text", text: "hello"}]}
      ]

      ctx = Context.new() |> Context.history(history) |> Context.build()

      assert ctx.messages == history
    end

    test "active tool result with tail_hints injects hint at distance 0 only" do
      # tail_hints only inject on the most recent turn (distance 0)
      history = [
        %{role: "user", content: "analyze this"},
        %{role: "assistant", content: [
          %{type: "tool_use", id: "t1", name: "hinted_tool", input: %{}}
        ]},
        %{role: "user", content: [
          %{type: "tool_result", tool_use_id: "t1", content: "some data"}
        ]}
      ]

      ctx = Context.new() |> Context.history(history) |> Context.age_tools(make_state(1)) |> Context.build()

      tool_result_msg = Enum.at(ctx.messages, 2)

      assert [
               %{type: "tool_result", content: "some data"},
               %{type: "text", text: "Focus on the key findings."}
             ] = tool_result_msg.content

      # When there's a later assistant message (distance > 0), no hints
      history_with_reply = history ++ [
        %{role: "assistant", content: [%{type: "text", text: "Here's my analysis"}]}
      ]

      ctx2 = Context.new() |> Context.history(history_with_reply) |> Context.age_tools(make_state(2)) |> Context.build()

      tool_result_msg2 = Enum.at(ctx2.messages, 2)

      assert [%{type: "tool_result", content: "some data"}] = tool_result_msg2.content
    end

    test "expired tool result does not include tail_hints" do
      history = [
        %{role: "user", content: "weather?"},
        %{role: "assistant", content: [
          %{type: "tool_use", id: "t1", name: "get_weather", input: %{"location" => "SF"}}
        ]},
        %{role: "user", content: [
          %{type: "tool_result", tool_use_id: "t1", content: "72°F"}
        ]},
        %{role: "assistant", content: [%{type: "text", text: "Nice!"}]},
        %{role: "user", content: "more"},
        %{role: "assistant", content: [%{type: "text", text: "ok"}]},
        %{role: "user", content: "again"},
        %{role: "assistant", content: [%{type: "text", text: "sure"}]}
      ]

      ctx = Context.new() |> Context.history(history) |> Context.age_tools(make_state(4)) |> Context.build()

      tool_result_msg = Enum.at(ctx.messages, 2)
      [result_block] = tool_result_msg.content

      # Should be expiry_message, no tail hint
      assert result_block.content =~ "Weather expired"
      refute Enum.any?(tool_result_msg.content, &match?(%{type: "text"}, &1))
    end
  end

  defp injected_documents(content) do
    Enum.flat_map(content, fn
      %Document{} = doc -> [doc]
      %{type: "document", document: %Document{} = doc} -> [doc]
      _ -> []
    end)
  end
end
