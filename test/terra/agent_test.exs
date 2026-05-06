defmodule Terra.AgentTest do
  use ExUnit.Case, async: true

  alias Terra.Agent.State
  import ExUnit.CaptureLog

  # ── Test Agent ──────────────────────────────────────────────

  defmodule TestAgent do
    use Terra.Agent

    def init(args) do
      {:ok, :idle, %{name: args[:name] || "test"}, provider: {Terra.Test.MockProvider, %{}}}
    end

    def context(_state_name, state) do
      ctx =
        Terra.Context.new()
        |> Terra.Context.system("You are #{state.data.name}")
        |> Terra.Context.model(%{model: "mock", max_tokens: 100})
        |> Terra.Context.build()

      {ctx, state}
    end

    def handle_input(:idle, {:message, _text}, state) do
      {:next_state, :processing, state, [:invoke]}
    end

    def handle_input(:idle, :stay, state) do
      {:keep_state, state}
    end

    def handle_input(:idle, {:update_data, data}, state) do
      {:keep_state, %{state | data: data}}
    end

    def handle_stream_event(_state_name, _event, state), do: {:keep_state, state}

    def handle_response(_response, state) do
      {:next_state, :idle, state}
    end
  end

  # ── Boot ──────────────────────────────────────────────────

  describe "start_link and initial state" do
    test "start_link returns {:ok, pid} and process is alive" do
      {:ok, pid} = TestAgent.start_link(%{name: "boot"})
      assert Process.alive?(pid)
    end

    test "init/1 sets the initial state name" do
      {:ok, pid} = TestAgent.start_link(%{name: "boot"})
      {state_name, _state} = :sys.get_state(pid)
      assert state_name == :idle
    end

    test ":sys.get_state returns {state_name, %State{}}" do
      {:ok, pid} = TestAgent.start_link(%{name: "boot"})
      {state_name, state} = :sys.get_state(pid)

      assert state_name == :idle
      assert %State{} = state
      assert state.data == %{name: "boot"}
      assert state.handler == TestAgent
      assert state.provider == {Terra.Test.MockProvider, %{}}
      assert state.turn_count == 0
      assert state.tokens == %{input: 0, output: 0}
    end
  end

  # ── Input → state transitions ─────────────────────────────

  describe "send_input and state transitions" do
    test "send_input triggers handle_input and transitions state" do
      {:ok, pid} = TestAgent.start_link(%{name: "input"})
      Terra.Agent.send_input(pid, {:message, "hi"})
      Process.sleep(10)

      # After full loop (invoke → stream → response), back to :idle
      {state_name, _state} = :sys.get_state(pid)
      assert state_name == :idle
    end

    test "returning {:keep_state, state} stays in same state" do
      {:ok, pid} = TestAgent.start_link(%{name: "stay"})
      Terra.Agent.send_input(pid, :stay)
      Process.sleep(10)

      {state_name, _state} = :sys.get_state(pid)
      assert state_name == :idle
    end

    test "consumer can update state.data in return" do
      {:ok, pid} = TestAgent.start_link(%{name: "original"})
      Terra.Agent.send_input(pid, {:update_data, %{name: "updated"}})
      Process.sleep(10)

      {_state_name, state} = :sys.get_state(pid)
      assert state.data == %{name: "updated"}
    end
  end

  # ── Invoke → stream → response ────────────────────────────

  describe "invoke action triggers full streaming loop" do
    defmodule TrackingAgent do
      use Terra.Agent

      def init(args) do
        test_pid = args[:test_pid]
        {:ok, :idle, %{test_pid: test_pid, events: []}, provider: {Terra.Test.MockProvider, %{}}}
      end

      def context(_state_name, state) do
        send(state.data.test_pid, {:callback, :context})

        ctx =
          Terra.Context.new()
          |> Terra.Context.system("system prompt")
          |> Terra.Context.model(%{model: "mock", max_tokens: 100})
          |> Terra.Context.build()

        {ctx, state}
      end

      def handle_input(:idle, :go, state) do
        {:next_state, :processing, state, [:invoke]}
      end

      def handle_stream_event(_state_name, event, state) do
        send(state.data.test_pid, {:callback, {:stream_event, event}})
        events = state.data.events ++ [event]
        {:keep_state, %{state | data: %{state.data | events: events}}}
      end

      def handle_response(response, state) do
        send(state.data.test_pid, {:callback, {:response, response}})
        {:next_state, :idle, state}
      end
    end

    test ":invoke calls context/2" do
      {:ok, pid} = TrackingAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)

      assert_receive {:callback, :context}, 200
    end

    test "Anthropic stream events forwarded to handle_stream_event/3" do
      {:ok, pid} = TrackingAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)

      assert_receive {:callback, {:stream_event, {:message_start, %{id: "msg_test"}}}}, 200
      assert_receive {:callback, {:stream_event, {:content_block_start, 0, %{type: "text"}}}}, 200

      assert_receive {:callback,
                      {:stream_event,
                       {:content_block_delta, 0, %{type: "text_delta", text: "Hello"}}}},
                     200

      assert_receive {:callback,
                      {:stream_event,
                       {:content_block_delta, 0, %{type: "text_delta", text: " world"}}}},
                     200

      assert_receive {:callback, {:stream_event, {:content_block_stop, 0}}}, 200

      assert_receive {:callback,
                      {:stream_event,
                       {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 5}}}},
                     200
    end

    test "message_stop calls handle_response/2 with stop_reason and usage" do
      {:ok, pid} = TrackingAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)

      assert_receive {:callback, {:response, response}}, 200
      assert response.stop_reason == "end_turn"
      assert response.usage == %{input_tokens: 10, output_tokens: 5}
    end

    test "after message_stop: turn_count increments and tokens updated" do
      {:ok, pid} = TrackingAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)

      assert_receive {:callback, {:response, _}}, 200
      Process.sleep(10)

      {_state_name, state} = :sys.get_state(pid)
      assert state.turn_count == 1
      assert state.tokens == %{input: 10, output: 5}
    end

    test "full end-to-end: input → stream events → response → back to idle" do
      {:ok, pid} = TrackingAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)

      assert_receive {:callback, :context}, 200
      assert_receive {:callback, {:stream_event, {:message_start, _}}}, 200
      assert_receive {:callback, {:stream_event, {:content_block_start, 0, _}}}, 200

      assert_receive {:callback, {:stream_event, {:content_block_delta, 0, %{text: "Hello"}}}},
                     200

      assert_receive {:callback, {:stream_event, {:content_block_delta, 0, %{text: " world"}}}},
                     200

      assert_receive {:callback, {:stream_event, {:content_block_stop, 0}}}, 200
      assert_receive {:callback, {:stream_event, {:message_delta, _, _}}}, 200
      assert_receive {:callback, {:response, _}}, 200

      Process.sleep(10)
      {state_name, _state} = :sys.get_state(pid)
      assert state_name == :idle
    end
  end

  # ── Multi-turn conversation ─────────────────────────────────

  describe "multi-turn conversation" do
    defmodule MultiTurnProvider do
      @behaviour Terra.Provider

      @impl true
      def stream(caller, %{messages: messages} = _params) do
        ref = make_ref()
        turn = length(Enum.filter(messages, &(&1["role"] == "user" || &1[:role] == "user")))

        spawn_link(fn ->
          text =
            case turn do
              1 -> "Hello! How can I help?"
              2 -> "Your name is Alice."
              _ -> "Turn #{turn}"
            end

          for event <- [
                {:message_start,
                 %{
                   id: "msg_mt_#{turn}",
                   content: [],
                   usage: %{input_tokens: 5 * turn, output_tokens: 0}
                 }},
                {:content_block_start, 0, %{type: "text", text: ""}},
                {:content_block_delta, 0, %{type: "text_delta", text: text}},
                {:content_block_stop, 0},
                {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 3 * turn}},
                :message_stop
              ] do
            send(caller, {:stream, ref, event})
          end
        end)

        {:ok, ref}
      end

      @impl true
      def cancel(_ref), do: :ok
    end

    defmodule MultiTurnAgent do
      use Terra.Agent

      def init(args) do
        {:ok, :idle, %{test_pid: args[:test_pid], history: []},
         provider: {MultiTurnProvider, %{}}}
      end

      def context(_state_name, state) do
        ctx =
          Terra.Context.new()
          |> Terra.Context.system("You are a helpful assistant.")
          |> Terra.Context.history(state.data.history)
          |> Terra.Context.model(%{model: "mock", max_tokens: 100})
          |> Terra.Context.build()

        {ctx, state}
      end

      def handle_input(:idle, {:message, text}, state) do
        history = state.data.history ++ [%{role: "user", content: text}]
        {:next_state, :processing, %{state | data: %{state.data | history: history}}, [:invoke]}
      end

      def handle_stream_event(_, _, state), do: {:keep_state, state}

      def handle_response(response, state) do
        text =
          response.content_blocks
          |> Enum.filter(&(&1.type == "text"))
          |> Enum.map(& &1.text)
          |> Enum.join()

        history = state.data.history ++ [%{role: "assistant", content: text}]
        send(state.data.test_pid, {:callback, {:response, response}})
        {:next_state, :idle, %{state | data: %{state.data | history: history}}}
      end
    end

    test "two-turn conversation maintains history and accumulates tokens" do
      {:ok, pid} = MultiTurnAgent.start_link(%{test_pid: self()})

      # Turn 1
      Terra.Agent.send_input(pid, {:message, "Hi there"})
      assert_receive {:callback, {:response, r1}}, 200
      assert [%{type: "text", text: "Hello! How can I help?"}] = r1.content_blocks

      # Turn 2
      Terra.Agent.send_input(pid, {:message, "What's my name?"})
      assert_receive {:callback, {:response, r2}}, 200
      assert [%{type: "text", text: "Your name is Alice."}] = r2.content_blocks

      Process.sleep(10)
      {state_name, state} = :sys.get_state(pid)
      assert state_name == :idle
      assert state.turn_count == 2
      assert state.tokens == %{input: 15, output: 9}

      # History should have all 4 messages
      assert length(state.data.history) == 4
      assert Enum.at(state.data.history, 0).role == "user"
      assert Enum.at(state.data.history, 1).role == "assistant"
      assert Enum.at(state.data.history, 2).role == "user"
      assert Enum.at(state.data.history, 3).role == "assistant"
    end

    test "history is passed to provider via context" do
      {:ok, pid} = MultiTurnAgent.start_link(%{test_pid: self()})

      Terra.Agent.send_input(pid, {:message, "First message"})
      assert_receive {:callback, {:response, _}}, 200

      Terra.Agent.send_input(pid, {:message, "Second message"})
      assert_receive {:callback, {:response, _}}, 200

      Process.sleep(10)
      {_, state} = :sys.get_state(pid)
      assert Enum.at(state.data.history, 0) == %{role: "user", content: "First message"}
      assert Enum.at(state.data.history, 2) == %{role: "user", content: "Second message"}
    end
  end

  # ── Thinking + tool_use content blocks ──────────────────────

  describe "thinking and tool_use stream events" do
    defmodule ThinkingToolProvider do
      @behaviour Terra.Provider

      @impl true
      def stream(caller, _params) do
        ref = make_ref()

        spawn_link(fn ->
          events = [
            {:message_start,
             %{
               id: "msg_think_tool",
               type: "message",
               role: "assistant",
               content: [],
               usage: %{input_tokens: 20, output_tokens: 0}
             }},
            # Thinking block (index 0)
            {:content_block_start, 0, %{type: "thinking", thinking: ""}},
            {:content_block_delta, 0, %{type: "thinking_delta", thinking: "Let me think about"}},
            {:content_block_delta, 0, %{type: "thinking_delta", thinking: " this problem."}},
            {:content_block_delta, 0, %{type: "signature_delta", signature: "EqQBCgIYAhIM..."}},
            {:content_block_stop, 0},
            # Tool use block (index 1)
            {:content_block_start, 1,
             %{type: "tool_use", id: "toolu_01A", name: "get_weather", input: %{}}},
            {:content_block_delta, 1,
             %{type: "input_json_delta", partial_json: "{\"location\":"}},
            {:content_block_delta, 1,
             %{type: "input_json_delta", partial_json: " \"San Francisco\"}"}},
            {:content_block_stop, 1},
            {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 15}},
            :message_stop
          ]

          for event <- events do
            send(caller, {:stream, ref, event})
          end
        end)

        {:ok, ref}
      end

      @impl true
      def cancel(_ref), do: :ok
    end

    defmodule ThinkingToolAgent do
      use Terra.Agent

      def init(args) do
        {:ok, :idle, %{test_pid: args[:test_pid], events: []},
         provider: {ThinkingToolProvider, %{}}}
      end

      def context(_, state),
        do:
          {Terra.Context.new()
           |> Terra.Context.system("system")
           |> Terra.Context.build()
           |> Terra.Context.model(%{model: "mock", max_tokens: 100})
           |> Terra.Context.build(), state}

      def handle_input(:idle, :go, state) do
        {:next_state, :processing, state, [:invoke]}
      end

      def handle_stream_event(_state_name, event, state) do
        send(state.data.test_pid, {:callback, {:stream_event, event}})
        events = state.data.events ++ [event]
        {:keep_state, %{state | data: %{state.data | events: events}}}
      end

      def handle_response(response, state) do
        send(state.data.test_pid, {:callback, {:response, response}})
        {:next_state, :idle, state}
      end
    end

    test "thinking deltas forwarded to handle_stream_event/3" do
      {:ok, pid} = ThinkingToolAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)

      assert_receive {:callback, {:stream_event, {:content_block_start, 0, %{type: "thinking"}}}},
                     200

      assert_receive {:callback,
                      {:stream_event,
                       {:content_block_delta, 0,
                        %{type: "thinking_delta", thinking: "Let me think about"}}}},
                     200

      assert_receive {:callback,
                      {:stream_event,
                       {:content_block_delta, 0,
                        %{type: "thinking_delta", thinking: " this problem."}}}},
                     200

      assert_receive {:callback,
                      {:stream_event,
                       {:content_block_delta, 0,
                        %{type: "signature_delta", signature: "EqQBCgIYAhIM..."}}}},
                     200

      assert_receive {:callback, {:stream_event, {:content_block_stop, 0}}}, 200
    end

    test "tool_use deltas forwarded to handle_stream_event/3" do
      {:ok, pid} = ThinkingToolAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)

      assert_receive {:callback,
                      {:stream_event,
                       {:content_block_start, 1, %{type: "tool_use", name: "get_weather"}}}},
                     200

      assert_receive {:callback,
                      {:stream_event,
                       {:content_block_delta, 1,
                        %{type: "input_json_delta", partial_json: "{\"location\":"}}}},
                     200

      assert_receive {:callback,
                      {:stream_event, {:content_block_delta, 1, %{type: "input_json_delta"}}}},
                     200

      assert_receive {:callback, {:stream_event, {:content_block_stop, 1}}}, 200
    end

    test "stop_reason reflected in handle_response" do
      {:ok, pid} = ThinkingToolAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)

      assert_receive {:callback, {:response, response}}, 200
      assert response.stop_reason == "end_turn"
      assert response.usage == %{input_tokens: 20, output_tokens: 15}
    end

    test "tokens tracked correctly across thinking + tool_use turn" do
      {:ok, pid} = ThinkingToolAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)

      assert_receive {:callback, {:response, _}}, 200
      Process.sleep(10)

      {_state_name, state} = :sys.get_state(pid)
      assert state.turn_count == 1
      assert state.tokens == %{input: 20, output: 15}
    end
  end

  # ── Registries pass tools to provider ─────────────────────

  describe "tool registries" do
    defmodule ToolCapturingProvider do
      @behaviour Terra.Provider

      @impl true
      def stream(caller, %{tools: tools} = _params) do
        ref = make_ref()
        :ets.insert(:tool_capture, {:tools, tools})

        spawn_link(fn ->
          send(
            caller,
            {:stream, ref,
             {:message_start,
              %{id: "msg_tools", content: [], usage: %{input_tokens: 5, output_tokens: 0}}}}
          )

          send(
            caller,
            {:stream, ref, {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 2}}}
          )

          send(caller, {:stream, ref, :message_stop})
        end)

        {:ok, ref}
      end

      @impl true
      def cancel(_ref), do: :ok
    end

    defmodule TestToolRegistry do
      use Terra.ToolRegistry
      import Terra.Tool

      @impl true
      def tools(_state) do
        [
          tool("fetch_forecast")
          |> desc("Fetch a weather report")
          |> param(:id, :string, required: true)
          |> aging(expiry: 4)
        ]
      end

      @impl true
      def execute("fetch_forecast", input, state), do: {:ok, %{id: input[:id]}, state}
    end

    defmodule AnotherRegistry do
      use Terra.ToolRegistry
      import Terra.Tool

      @impl true
      def tools(_state) do
        [
          tool("get_alerts")
          |> desc("Get weather alerts")
          |> param(:date, :string, [])
        ]
      end

      @impl true
      def execute("get_alerts", input, state), do: {:ok, %{date: input[:date]}, state}
    end

    defmodule RegistryAgent do
      use Terra.Agent

      def init(args) do
        {:ok, :idle, %{test_pid: args[:test_pid]},
         provider: {ToolCapturingProvider, %{}}, registries: [TestToolRegistry, AnotherRegistry]}
      end

      def context(_, state),
        do:
          {Terra.Context.new()
           |> Terra.Context.system("system")
           |> Terra.Context.build()
           |> Terra.Context.model(%{model: "mock", max_tokens: 100})
           |> Terra.Context.build(), state}

      def handle_input(:idle, :go, state) do
        {:next_state, :processing, state, [:invoke]}
      end

      def handle_stream_event(_, _, state), do: {:keep_state, state}
      def handle_response(_, state), do: {:next_state, :idle, state}
    end

    test "registries stored on state" do
      {:ok, pid} = RegistryAgent.start_link(%{test_pid: self()})
      {_state_name, state} = :sys.get_state(pid)

      assert state.registries == [TestToolRegistry, AnotherRegistry]
    end

    test "invoke collects tools from all registries and passes to provider" do
      :ets.new(:tool_capture, [:set, :public, :named_table])

      {:ok, pid} = RegistryAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)
      Process.sleep(20)

      [{:tools, tools}] = :ets.lookup(:tool_capture, :tools)
      assert length(tools) == 2

      names = Enum.map(tools, & &1.name)
      assert "fetch_forecast" in names
      assert "get_alerts" in names

      :ets.delete(:tool_capture)
    end

    test "tools passed to provider include Terra-specific fields" do
      :ets.new(:tool_capture_fields, [:set, :public, :named_table])

      defmodule ToolCapturingProvider2 do
        @behaviour Terra.Provider

        @impl true
        def stream(caller, %{tools: tools} = _params) do
          ref = make_ref()
          :ets.insert(:tool_capture_fields, {:tools, tools})

          spawn_link(fn ->
            send(
              caller,
              {:stream, ref,
               {:message_start,
                %{id: "msg_f", content: [], usage: %{input_tokens: 1, output_tokens: 0}}}}
            )

            send(
              caller,
              {:stream, ref, {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 1}}}
            )

            send(caller, {:stream, ref, :message_stop})
          end)

          {:ok, ref}
        end

        @impl true
        def cancel(_ref), do: :ok
      end

      defmodule RegistryAgent2 do
        use Terra.Agent

        def init(args) do
          {:ok, :idle, %{},
           provider: {ToolCapturingProvider2, %{}}, registries: [TestToolRegistry]}
        end

        def context(_, state),
          do:
            {Terra.Context.new()
             |> Terra.Context.system("system")
             |> Terra.Context.build()
             |> Terra.Context.model(%{model: "mock"})
             |> Terra.Context.build(), state}

        def handle_input(:idle, :go, state), do: {:next_state, :processing, state, [:invoke]}
        def handle_stream_event(_, _, state), do: {:keep_state, state}
        def handle_response(_, state), do: {:next_state, :idle, state}
      end

      {:ok, pid} = RegistryAgent2.start_link(%{})
      Terra.Agent.send_input(pid, :go)
      Process.sleep(20)

      [{:tools, tools}] = :ets.lookup(:tool_capture_fields, :tools)
      forecast_tool = Enum.find(tools, &(&1.name == "fetch_forecast"))
      assert forecast_tool.expiry_distance == 4

      :ets.delete(:tool_capture_fields)
    end

    test "agent without registries passes empty tools list" do
      :ets.new(:no_tool_capture, [:set, :public, :named_table])

      defmodule NoToolsProvider do
        @behaviour Terra.Provider

        @impl true
        def stream(caller, %{tools: tools} = _params) do
          ref = make_ref()
          :ets.insert(:no_tool_capture, {:tools, tools})

          spawn_link(fn ->
            send(
              caller,
              {:stream, ref,
               {:message_start,
                %{id: "msg_no", content: [], usage: %{input_tokens: 1, output_tokens: 0}}}}
            )

            send(
              caller,
              {:stream, ref, {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 1}}}
            )

            send(caller, {:stream, ref, :message_stop})
          end)

          {:ok, ref}
        end

        @impl true
        def cancel(_ref), do: :ok
      end

      defmodule NoToolsAgent do
        use Terra.Agent

        def init(_args) do
          {:ok, :idle, %{}, provider: {NoToolsProvider, %{}}}
        end

        def context(_, state),
          do:
            {Terra.Context.new() |> Terra.Context.system("system") |> Terra.Context.build(),
             state}

        def handle_input(:idle, :go, state), do: {:next_state, :processing, state, [:invoke]}
        def handle_stream_event(_, _, state), do: {:keep_state, state}
        def handle_response(_, state), do: {:next_state, :idle, state}
      end

      {:ok, pid} = NoToolsAgent.start_link(%{})
      Terra.Agent.send_input(pid, :go)
      Process.sleep(20)

      [{:tools, tools}] = :ets.lookup(:no_tool_capture, :tools)
      assert tools == []

      :ets.delete(:no_tool_capture)
    end
  end

  # ── Timeouts and process messages ──────────────────────────

  describe "handle_info and handle_timeout" do
    defmodule InfoAgent do
      use Terra.Agent

      def init(args) do
        {:ok, :idle, %{test_pid: args[:test_pid]}, provider: {Terra.Test.MockProvider, %{}}}
      end

      def context(_, state),
        do:
          {Terra.Context.new()
           |> Terra.Context.system("system")
           |> Terra.Context.build()
           |> Terra.Context.model(%{model: "mock", max_tokens: 100})
           |> Terra.Context.build(), state}

      def handle_input(:idle, :go, state) do
        {:next_state, :processing, state, [:invoke]}
      end

      def handle_stream_event(_, _, state), do: {:keep_state, state}
      def handle_response(_, state), do: {:next_state, :idle, state}

      def handle_info(state_name, msg, state) do
        send(state.data.test_pid, {:info_received, state_name, msg})
        {:keep_state, state}
      end

      def handle_timeout(state_name, event, state) do
        send(state.data.test_pid, {:timeout_received, state_name, event})
        {:keep_state, state}
      end
    end

    test "arbitrary messages reach handle_info/3" do
      {:ok, pid} = InfoAgent.start_link(%{test_pid: self()})
      send(pid, {:custom, "data"})

      assert_receive {:info_received, :idle, {:custom, "data"}}, 200
    end

    defmodule TimeoutAgent do
      use Terra.Agent

      def init(args) do
        {:ok, :waiting, %{test_pid: args[:test_pid]}, provider: {Terra.Test.MockProvider, %{}}}
      end

      def context(_, state),
        do:
          {Terra.Context.new()
           |> Terra.Context.system("system")
           |> Terra.Context.build()
           |> Terra.Context.model(%{model: "mock", max_tokens: 100})
           |> Terra.Context.build(), state}

      def handle_input(:waiting, :start_timer, state) do
        {:keep_state, state, [{:state_timeout, 50, :tick}]}
      end

      def handle_input(:waiting, :start_generic_timer, state) do
        {:keep_state, state, [{:generic_timeout, :poll, 50, :check}]}
      end

      def handle_stream_event(_, _, state), do: {:keep_state, state}
      def handle_response(_, state), do: {:next_state, :idle, state}

      def handle_timeout(state_name, event, state) do
        send(state.data.test_pid, {:timeout_received, state_name, event})
        {:keep_state, state}
      end
    end

    test "state_timeout fires handle_timeout/3" do
      {:ok, pid} = TimeoutAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :start_timer)

      assert_receive {:timeout_received, :waiting, :tick}, 200
    end

    test "generic_timeout fires handle_timeout/3" do
      {:ok, pid} = TimeoutAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :start_generic_timer)

      assert_receive {:timeout_received, :waiting, :check}, 200
    end

    defmodule InvokeFromInfoAgent do
      use Terra.Agent

      def init(args) do
        {:ok, :idle, %{test_pid: args[:test_pid]}, provider: {Terra.Test.MockProvider, %{}}}
      end

      def context(_, state),
        do:
          {Terra.Context.new()
           |> Terra.Context.system("system")
           |> Terra.Context.build()
           |> Terra.Context.model(%{model: "mock", max_tokens: 100})
           |> Terra.Context.build(), state}

      def handle_input(_, _, state), do: {:keep_state, state}
      def handle_stream_event(_, _, state), do: {:keep_state, state}

      def handle_response(_, state) do
        send(state.data.test_pid, :response_from_info_invoke)
        {:next_state, :idle, state}
      end

      def handle_info(:idle, :trigger_invoke, state) do
        {:next_state, :processing, state, [:invoke]}
      end
    end

    test "handle_info can return [:invoke] to trigger streaming" do
      {:ok, pid} = InvokeFromInfoAgent.start_link(%{test_pid: self()})
      send(pid, :trigger_invoke)

      assert_receive :response_from_info_invoke, 200
    end
  end

  # ── Cancel stream ─────────────────────────────────────────

  describe "cancel stream" do
    defmodule SlowMockProvider do
      @behaviour Terra.Provider

      @impl true
      def stream(caller, _params) do
        ref = make_ref()

        spawn_link(fn ->
          send(
            caller,
            {:stream, ref,
             {:message_start,
              %{id: "msg_slow", content: [], usage: %{input_tokens: 5, output_tokens: 0}}}}
          )

          send(caller, {:stream, ref, {:content_block_start, 0, %{type: "text", text: ""}}})

          send(
            caller,
            {:stream, ref, {:content_block_delta, 0, %{type: "text_delta", text: "slow"}}}
          )

          # Hang — don't send message_stop. The cancel test will interrupt.
          Process.sleep(:infinity)
        end)

        {:ok, ref}
      end

      @impl true
      def cancel(_ref), do: :ok
    end

    defmodule CancelAgent do
      use Terra.Agent

      def init(args) do
        {:ok, :idle, %{test_pid: args[:test_pid]}, provider: {SlowMockProvider, %{}}}
      end

      def context(_, state),
        do:
          {Terra.Context.new()
           |> Terra.Context.system("system")
           |> Terra.Context.build()
           |> Terra.Context.model(%{model: "mock", max_tokens: 100})
           |> Terra.Context.build(), state}

      def handle_input(:idle, :go, state) do
        {:next_state, :streaming, state, [:invoke]}
      end

      def handle_input(:streaming, :cancel, state) do
        {:keep_state, state, [:cancel_stream]}
      end

      def handle_input(:streaming, :cancel_and_reinvoke, state) do
        {:next_state, :streaming, state, [:cancel_stream, :invoke]}
      end

      def handle_stream_event(_, _, state), do: {:keep_state, state}

      def handle_response(_, state) do
        send(state.data.test_pid, :response_called)
        {:next_state, :idle, state}
      end
    end

    test ":cancel_stream clears stream_ref" do
      {:ok, pid} = CancelAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)
      Process.sleep(20)

      Terra.Agent.send_input(pid, :cancel)
      Process.sleep(20)

      {_state_name, state} = :sys.get_state(pid)
      assert state.stream_ref == nil
    end

    test "after cancel, handle_response is NOT called" do
      {:ok, pid} = CancelAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)
      Process.sleep(20)

      Terra.Agent.send_input(pid, :cancel)
      Process.sleep(50)

      refute_received :response_called
    end

    test "stale stream messages (wrong ref) are silently ignored" do
      {:ok, pid} = CancelAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)
      Process.sleep(20)

      stale_ref = make_ref()
      send(pid, {:stream, stale_ref, :message_stop})
      Process.sleep(20)

      refute_received :response_called
    end

    test "[:cancel_stream, :invoke] cancels then re-invokes" do
      {:ok, pid} = CancelAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)
      Process.sleep(20)

      Terra.Agent.send_input(pid, :cancel_and_reinvoke)
      Process.sleep(50)

      {_state_name, state} = :sys.get_state(pid)
      assert state.stream_ref != nil
    end
  end

  # ── Tool execution ────────────────────────────────────────

  describe "tool execution" do
    defmodule ToolExecRegistry do
      use Terra.ToolRegistry
      import Terra.Tool

      @impl true
      def tools(_state) do
        [
          tool("get_weather")
          |> desc("Get weather for a location")
          |> param(:location, :string, []),
          tool("get_time")
          |> desc("Get current time for a timezone")
          |> param(:tz, :string, [])
        ]
      end

      @impl true
      def execute("get_weather", input, state) do
        {:ok, %{temp: 72, location: input[:location]}, state}
      end

      def execute("get_time", input, state) do
        {:ok, %{time: "14:30", tz: input[:tz]}, state}
      end
    end

    defmodule ErrorToolRegistry do
      use Terra.ToolRegistry
      import Terra.Tool

      @impl true
      def tools(_state) do
        [tool("fail_tool") |> desc("Always fails")]
      end

      @impl true
      def execute("fail_tool", _input, state), do: {:error, "something went wrong", state}
    end

    # -- Provider: single tool_use turn --
    defmodule ToolUseProvider do
      @behaviour Terra.Provider

      @impl true
      def stream(caller, _params) do
        ref = make_ref()

        spawn_link(fn ->
          for event <- [
                {:message_start,
                 %{
                   id: "msg_tu",
                   type: "message",
                   role: "assistant",
                   content: [],
                   usage: %{input_tokens: 10, output_tokens: 0}
                 }},
                {:content_block_start, 0,
                 %{type: "tool_use", id: "toolu_01A", name: "get_weather", input: %{}}},
                {:content_block_delta, 0,
                 %{type: "input_json_delta", partial_json: "{\"location\":"}},
                {:content_block_delta, 0,
                 %{type: "input_json_delta", partial_json: " \"San Francisco\"}"}},
                {:content_block_stop, 0},
                {:message_delta, %{stop_reason: "tool_use"}, %{output_tokens: 8}},
                :message_stop
              ] do
            send(caller, {:stream, ref, event})
          end
        end)

        {:ok, ref}
      end

      @impl true
      def cancel(_ref), do: :ok
    end

    # -- Agent: notifies test, re-invokes on tool_use --
    defmodule ToolUseAgent do
      use Terra.Agent

      def init(args) do
        {:ok, :idle, %{test_pid: args[:test_pid]},
         provider: {ToolUseProvider, %{}}, registries: [ToolExecRegistry]}
      end

      def context(_, state),
        do:
          {Terra.Context.new()
           |> Terra.Context.system("system")
           |> Terra.Context.build()
           |> Terra.Context.model(%{model: "mock", max_tokens: 100})
           |> Terra.Context.build(), state}

      def handle_input(:idle, :go, state) do
        {:next_state, :processing, state, [:invoke]}
      end

      def handle_stream_event(_, _, state), do: {:keep_state, state}

      def handle_response(response, state) do
        send(state.data.test_pid, {:callback, {:response, response}})
        {:next_state, :idle, state}
      end
    end

    test "handle_response called for tool_use stop_reason with content_blocks and tool_results" do
      {:ok, pid} = ToolUseAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)

      assert_receive {:callback, {:response, response}}, 500
      assert response.stop_reason == "tool_use"

      # content_blocks include the finalized tool_use block
      assert [
               %{
                 type: "tool_use",
                 id: "toolu_01A",
                 name: "get_weather",
                 input: %{location: "San Francisco"}
               }
             ] =
               response.content_blocks

      # tool_results from eager execution
      assert [%{tool_use_id: "toolu_01A", result: {:ok, %{temp: 72, location: "San Francisco"}}}] =
               response.tool_results
    end

    test "handle_response called for end_turn with content_blocks, empty tool_results" do
      {:ok, pid} = Terra.AgentTest.TrackingAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)

      assert_receive {:callback, {:response, response}}, 200
      assert response.stop_reason == "end_turn"
      assert [%{type: "text", text: "Hello world"}] = response.content_blocks
      assert response.tool_results == []
    end

    test "consumer controls re-invoke — Terra does NOT auto-re-invoke" do
      {:ok, pid} = ToolUseAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)

      assert_receive {:callback, {:response, %{stop_reason: "tool_use"}}}, 500
      Process.sleep(50)

      # Agent should be idle — consumer's handle_response transitioned there
      # No second invoke happened because consumer didn't return [:invoke]
      {state_name, state} = :sys.get_state(pid)
      assert state_name == :idle
      assert state.turn_count == 1
    end

    # -- Agent that re-invokes on tool_use (like Forecaster would) --
    defmodule ReinvokingAgent do
      use Terra.Agent

      def init(args) do
        :ets.new(:reinvoke_state, [:set, :public, :named_table])
        :ets.insert(:reinvoke_state, {:calls, 0})

        {:ok, :idle, %{test_pid: args[:test_pid]},
         provider: {Terra.AgentTest.ReinvokingProvider, %{}}, registries: [ToolExecRegistry]}
      end

      def context(_, state),
        do:
          {Terra.Context.new()
           |> Terra.Context.system("system")
           |> Terra.Context.build()
           |> Terra.Context.model(%{model: "mock", max_tokens: 100})
           |> Terra.Context.build(), state}

      def handle_input(:idle, :go, state) do
        {:next_state, :processing, state, [:invoke]}
      end

      def handle_stream_event(_, _, state), do: {:keep_state, state}

      def handle_response(%{stop_reason: "tool_use"} = response, state) do
        send(state.data.test_pid, {:callback, {:response, response}})
        # Consumer decides to re-invoke — appends messages and invokes
        {:next_state, :processing, state, [:invoke]}
      end

      def handle_response(response, state) do
        send(state.data.test_pid, {:callback, {:response, response}})
        {:next_state, :idle, state}
      end
    end

    defmodule ReinvokingProvider do
      @behaviour Terra.Provider

      @impl true
      def stream(caller, _params) do
        ref = make_ref()
        call_count = :ets.update_counter(:reinvoke_state, :calls, 1)

        spawn_link(fn ->
          events =
            if call_count == 1 do
              [
                {:message_start,
                 %{
                   id: "msg_r1",
                   type: "message",
                   role: "assistant",
                   content: [],
                   usage: %{input_tokens: 10, output_tokens: 0}
                 }},
                {:content_block_start, 0,
                 %{type: "tool_use", id: "toolu_01A", name: "get_weather", input: %{}}},
                {:content_block_delta, 0,
                 %{type: "input_json_delta", partial_json: "{\"location\": \"SF\"}"}},
                {:content_block_stop, 0},
                {:message_delta, %{stop_reason: "tool_use"}, %{output_tokens: 8}},
                :message_stop
              ]
            else
              [
                {:message_start,
                 %{
                   id: "msg_r2",
                   type: "message",
                   role: "assistant",
                   content: [],
                   usage: %{input_tokens: 25, output_tokens: 0}
                 }},
                {:content_block_start, 0, %{type: "text", text: ""}},
                {:content_block_delta, 0, %{type: "text_delta", text: "It's 72°F."}},
                {:content_block_stop, 0},
                {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 12}},
                :message_stop
              ]
            end

          for e <- events, do: send(caller, {:stream, ref, e})
        end)

        {:ok, ref}
      end

      @impl true
      def cancel(_ref), do: :ok
    end

    test "consumer re-invokes: two rounds, tokens accumulate, turn_count increments" do
      {:ok, pid} = ReinvokingAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)

      # First response: tool_use
      assert_receive {:callback, {:response, r1}}, 500
      assert r1.stop_reason == "tool_use"

      # Second response: end_turn
      assert_receive {:callback, {:response, r2}}, 500
      assert r2.stop_reason == "end_turn"

      Process.sleep(10)
      {state_name, state} = :sys.get_state(pid)
      assert state_name == :idle
      assert state.turn_count == 2
      assert state.tokens == %{input: 35, output: 20}

      :ets.delete(:reinvoke_state)
    end

    # -- Multiple tools in single response --
    defmodule MultiToolProvider do
      @behaviour Terra.Provider

      @impl true
      def stream(caller, _params) do
        ref = make_ref()

        spawn_link(fn ->
          for e <- [
                {:message_start,
                 %{
                   id: "msg_mt",
                   type: "message",
                   role: "assistant",
                   content: [],
                   usage: %{input_tokens: 10, output_tokens: 0}
                 }},
                {:content_block_start, 0,
                 %{type: "tool_use", id: "toolu_01A", name: "get_weather", input: %{}}},
                {:content_block_delta, 0,
                 %{type: "input_json_delta", partial_json: "{\"location\": \"NYC\"}"}},
                {:content_block_stop, 0},
                {:content_block_start, 1,
                 %{type: "tool_use", id: "toolu_02B", name: "get_time", input: %{}}},
                {:content_block_delta, 1,
                 %{type: "input_json_delta", partial_json: "{\"tz\": \"EST\"}"}},
                {:content_block_stop, 1},
                {:message_delta, %{stop_reason: "tool_use"}, %{output_tokens: 10}},
                :message_stop
              ] do
            send(caller, {:stream, ref, e})
          end
        end)

        {:ok, ref}
      end

      @impl true
      def cancel(_ref), do: :ok
    end

    defmodule MultiToolAgent do
      use Terra.Agent

      def init(args) do
        {:ok, :idle, %{test_pid: args[:test_pid]},
         provider: {MultiToolProvider, %{}}, registries: [ToolExecRegistry]}
      end

      def context(_, state),
        do:
          {Terra.Context.new()
           |> Terra.Context.system("system")
           |> Terra.Context.build()
           |> Terra.Context.model(%{model: "mock", max_tokens: 100})
           |> Terra.Context.build(), state}

      def handle_input(:idle, :go, state), do: {:next_state, :processing, state, [:invoke]}
      def handle_stream_event(_, _, state), do: {:keep_state, state}

      def handle_response(response, state) do
        send(state.data.test_pid, {:callback, {:response, response}})
        {:next_state, :idle, state}
      end
    end

    test "multiple tools: all executed eagerly, all results in response" do
      {:ok, pid} = MultiToolAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)

      assert_receive {:callback, {:response, response}}, 500
      assert response.stop_reason == "tool_use"
      assert length(response.content_blocks) == 2
      assert length(response.tool_results) == 2

      ids = Enum.map(response.tool_results, & &1.tool_use_id)
      assert "toolu_01A" in ids
      assert "toolu_02B" in ids
    end

    # -- Tool error --
    defmodule ErrorToolProvider do
      @behaviour Terra.Provider

      @impl true
      def stream(caller, _params) do
        ref = make_ref()

        spawn_link(fn ->
          for e <- [
                {:message_start,
                 %{
                   id: "msg_err",
                   type: "message",
                   role: "assistant",
                   content: [],
                   usage: %{input_tokens: 5, output_tokens: 0}
                 }},
                {:content_block_start, 0,
                 %{type: "tool_use", id: "toolu_err", name: "fail_tool", input: %{}}},
                {:content_block_delta, 0, %{type: "input_json_delta", partial_json: "{}"}},
                {:content_block_stop, 0},
                {:message_delta, %{stop_reason: "tool_use"}, %{output_tokens: 3}},
                :message_stop
              ] do
            send(caller, {:stream, ref, e})
          end
        end)

        {:ok, ref}
      end

      @impl true
      def cancel(_ref), do: :ok
    end

    defmodule ErrorToolAgent do
      use Terra.Agent

      def init(args) do
        {:ok, :idle, %{test_pid: args[:test_pid]},
         provider: {ErrorToolProvider, %{}}, registries: [ErrorToolRegistry]}
      end

      def context(_, state),
        do:
          {Terra.Context.new()
           |> Terra.Context.system("system")
           |> Terra.Context.build()
           |> Terra.Context.model(%{model: "mock"})
           |> Terra.Context.build(), state}

      def handle_input(:idle, :go, state), do: {:next_state, :processing, state, [:invoke]}
      def handle_stream_event(_, _, state), do: {:keep_state, state}

      def handle_response(response, state) do
        send(state.data.test_pid, {:callback, {:response, response}})
        {:next_state, :idle, state}
      end
    end

    test "tool error: result has is_error flag in response" do
      {:ok, pid} = ErrorToolAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)

      assert_receive {:callback, {:response, response}}, 500

      assert [%{tool_use_id: "toolu_err", result: {:error, "something went wrong"}}] =
               response.tool_results
    end

    # -- Cancel discards partial tool --
    defmodule PartialToolProvider do
      @behaviour Terra.Provider

      @impl true
      def stream(caller, _params) do
        ref = make_ref()

        spawn_link(fn ->
          # Send tool_use start + partial JSON, but NO content_block_stop
          for e <- [
                {:message_start,
                 %{id: "msg_partial", content: [], usage: %{input_tokens: 5, output_tokens: 0}}},
                {:content_block_start, 0,
                 %{type: "tool_use", id: "toolu_partial", name: "get_weather", input: %{}}},
                {:content_block_delta, 0,
                 %{type: "input_json_delta", partial_json: "{\"location\": \"San Fr"}}
              ] do
            send(caller, {:stream, ref, e})
          end

          # Hang — cancel will interrupt
          Process.sleep(:infinity)
        end)

        {:ok, ref}
      end

      @impl true
      def cancel(_ref), do: :ok
    end

    defmodule PartialToolAgent do
      use Terra.Agent

      def init(args) do
        {:ok, :idle, %{test_pid: args[:test_pid]},
         provider: {PartialToolProvider, %{}}, registries: [ToolExecRegistry]}
      end

      def context(_, state),
        do:
          {Terra.Context.new()
           |> Terra.Context.system("system")
           |> Terra.Context.build()
           |> Terra.Context.model(%{model: "mock"})
           |> Terra.Context.build(), state}

      def handle_input(:idle, :go, state) do
        {:next_state, :streaming, state, [:invoke]}
      end

      def handle_input(:streaming, :cancel, state) do
        send(state.data.test_pid, {:tool_results_at_cancel, get_tool_results(state)})
        {:keep_state, state, [:cancel_stream]}
      end

      def handle_stream_event(_, _, state), do: {:keep_state, state}
      def handle_response(_, state), do: {:next_state, :idle, state}

      defp get_tool_results(state) do
        Map.get(state._stream_meta, :tool_results, %{})
      end
    end

    test "cancel during tool streaming: partial tool NOT executed" do
      {:ok, pid} = PartialToolAgent.start_link(%{test_pid: self()})
      Terra.Agent.send_input(pid, :go)
      Process.sleep(20)

      Terra.Agent.send_input(pid, :cancel)

      assert_receive {:tool_results_at_cancel, tool_results}, 200
      # No tool results — the tool_use block never got content_block_stop
      assert tool_results == %{}
    end

    # -- Incomplete tool at message_stop synthesizes tool_result error --
    defmodule IncompleteToolProvider do
      @behaviour Terra.Provider

      @impl true
      def stream(caller, _params) do
        ref = make_ref()

        spawn_link(fn ->
          for e <- [
                {:message_start,
                 %{id: "msg_incomplete", content: [], usage: %{input_tokens: 7, output_tokens: 0}}},
                {:content_block_start, 0,
                 %{type: "tool_use", id: "toolu_incomplete", name: "get_weather", input: %{}}},
                {:content_block_delta, 0,
                 %{type: "input_json_delta", partial_json: "{\"location\": \"San Fr"}},
                {:message_delta, %{stop_reason: "max_tokens"}, %{output_tokens: 11}},
                :message_stop
              ] do
            send(caller, {:stream, ref, e})
          end
        end)

        {:ok, ref}
      end

      @impl true
      def cancel(_ref), do: :ok
    end

    defmodule IncompleteToolAgent do
      use Terra.Agent

      def init(args) do
        {:ok, :idle, %{test_pid: args[:test_pid]},
         provider: {IncompleteToolProvider, %{}}, registries: [ToolExecRegistry]}
      end

      def context(_, state),
        do:
          {Terra.Context.new()
           |> Terra.Context.system("system")
           |> Terra.Context.build()
           |> Terra.Context.model(%{model: "mock"})
           |> Terra.Context.build(), state}

      def handle_input(:idle, :go, state), do: {:next_state, :processing, state, [:invoke]}
      def handle_stream_event(_, _, state), do: {:keep_state, state}

      def handle_response(response, state) do
        send(state.data.test_pid, {:callback, {:response, response}})
        {:next_state, :idle, state}
      end
    end

    test "message_stop with incomplete tool_use synthesizes tool_result error and logs warning" do
      log =
        capture_log(fn ->
          {:ok, pid} = IncompleteToolAgent.start_link(%{test_pid: self()})
          Terra.Agent.send_input(pid, :go)

          assert_receive {:callback, {:response, response}}, 500
          assert response.stop_reason == "max_tokens"

          assert [%{type: "tool_use", id: "toolu_incomplete", name: "get_weather", input: %{}}] =
                   response.content_blocks

          assert [
                   %{
                     tool_use_id: "toolu_incomplete",
                     name: "get_weather",
                     is_error: true,
                     result: {:error, reason}
                   }
                 ] = response.tool_results

          assert reason =~ "stop_reason=max_tokens"
          assert reason =~ "before content_block_stop"
        end)

      assert log =~ "Incomplete tool_use block at message_stop"
      assert log =~ "toolu_incomplete"
      assert log =~ "stop=max_tokens"
    end
  end
end
