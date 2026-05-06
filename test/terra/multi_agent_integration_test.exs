defmodule Terra.MultiAgentIntegrationTest do
  @moduledoc """
  Integration test simulating a morpheus-style multi-agent session.

  Three agents share a kernel:
  - **Conversation**: Main agent — talks to user, reads kernel buffers as context
  - **Planner**: Background — produces action plans, writes to :plan
  - **Analysis**: Background — generates insights, writes to :analysis

  Exercises: kernel read/write, locking, cross-agent coordination via
  notifications, queue + cooldown pattern, document injection from kernel,
  rest_for_one supervisor restart.
  """
  use ExUnit.Case, async: true

  alias Terra.Kernel
  alias Terra.Document

  # ── Providers ───────────────────────────────────────────

  defmodule ConversationProvider do
    @behaviour Terra.Provider

    @impl true
    def stream(caller, params) do
      ref = make_ref()
      table = params[:_test_table]

      call_count =
        if table do
          :ets.update_counter(table, :conv_calls, 1)
        else
          1
        end

      spawn_link(fn ->
        events =
          case table && :ets.lookup(table, {:conv_events, call_count}) do
            [{_, evts}] -> evts
            _ -> default_text_events("msg_conv_#{call_count}", "I can help with that.", 30, 15)
          end

        for e <- events, do: send(caller, {:stream, ref, e})
      end)

      {:ok, ref}
    end

    @impl true
    def cancel(_ref), do: :ok

    defp default_text_events(id, text, input, output) do
      [
        {:message_start, %{id: id, content: [], usage: %{input_tokens: input, output_tokens: 0}}},
        {:content_block_start, 0, %{type: "text", text: ""}},
        {:content_block_delta, 0, %{type: "text_delta", text: text}},
        {:content_block_stop, 0},
        {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: output}},
        :message_stop
      ]
    end
  end

  defmodule PlannerProvider do
    @behaviour Terra.Provider

    @impl true
    def stream(caller, params) do
      ref = make_ref()

      spawn_link(fn ->
        # The "planned" content echoes what we pass in the system prompt
        text = params[:system] || "No context"

        events = [
          {:message_start, %{id: "msg_plan", content: [], usage: %{input_tokens: 20, output_tokens: 0}}},
          {:content_block_start, 0, %{type: "text", text: ""}},
          {:content_block_delta, 0, %{type: "text_delta", text: "Plan: #{text}"}},
          {:content_block_stop, 0},
          {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 10}},
          :message_stop
        ]

        for e <- events, do: send(caller, {:stream, ref, e})
      end)

      {:ok, ref}
    end

    @impl true
    def cancel(_ref), do: :ok
  end

  defmodule AnalysisProvider do
    @behaviour Terra.Provider

    @impl true
    def stream(caller, _params) do
      ref = make_ref()

      spawn_link(fn ->
        events = [
          {:message_start, %{id: "msg_analysis", content: [], usage: %{input_tokens: 25, output_tokens: 0}}},
          {:content_block_start, 0, %{type: "text", text: ""}},
          {:content_block_delta, 0, %{type: "text_delta", text: "User shows interest in outdoor activity recommendations."}},
          {:content_block_stop, 0},
          {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 12}},
          :message_stop
        ]

        for e <- events, do: send(caller, {:stream, ref, e})
      end)

      {:ok, ref}
    end

    @impl true
    def cancel(_ref), do: :ok
  end

  # ── Conversation Agent ──────────────────────────────────

  defmodule ConvAgent do
    use Terra.Agent

    def init(args) do
      data = %{
        test_pid: args[:test_pid],
        interactions: [],
        active_buffer: "",
        provider_config: args[:provider_config] || %{}
      }

      {:ok, :idle, data,
       provider: {ConversationProvider, data.provider_config},
       kernel: args[:kernel]}
    end

    def context(_, state) do
      # Read kernel buffers — nil-safe document injection
      plan = Kernel.read(state.kernel, :plan)
      analysis = Kernel.read(state.kernel, :analysis)

      ctx =
        Terra.Context.new()
        |> Terra.Context.system("You are a helpful planning assistant.")
        |> Terra.Context.document(plan)
        |> Terra.Context.document(analysis)
        |> Terra.Context.history(state.data.interactions)
        |> Terra.Context.model(%{
          model: "mock",
          _test_table: state.data.provider_config[:table]
        })
        |> Terra.Context.age_tools(state)
        |> Terra.Context.build()

      {ctx, state}
    end

    def handle_input(:idle, {:message, text}, state) do
      state = append(state, "user", text)
      notify(state, {:conv, :processing})
      {:next_state, :processing, state, [:invoke]}
    end

    def handle_stream_event(:processing, {:message_start, _}, state) do
      {:next_state, :streaming, state}
    end

    def handle_stream_event(_, {:content_block_delta, _, %{type: "text_delta", text: text}}, state) do
      state = %{state | data: %{state.data | active_buffer: state.data.active_buffer <> text}}
      {:keep_state, state}
    end

    def handle_stream_event(_, _, state), do: {:keep_state, state}

    def handle_response(response, state) do
      state = append(state, "assistant", response.content_blocks)
      buffer = state.data.active_buffer
      state = %{state | data: %{state.data | active_buffer: ""}}

      notify(state, {:conv, :response, buffer})

      # Notify background agents about the new turn
      notify_background_agents(state)

      {:next_state, :idle, state}
    end

    # Listen for kernel updates
    def handle_info(_, {:kernel_update, slot, _doc}, state) do
      notify(state, {:conv, :kernel_updated, slot})
      {:keep_state, state}
    end

    def handle_info(_, _, state), do: {:keep_state, state}

    defp append(state, role, content) do
      interaction = %{role: role, content: content}
      %{state | data: %{state.data | interactions: state.data.interactions ++ [interaction]}}
    end

    defp notify(state, event), do: send(state.data.test_pid, {:agent, event})

    defp notify_background_agents(state) do
      # In real code: :gen_statem.cast to planner/analysis agents via Registry
      # For testing: send to test_pid which orchestrates
      send(state.data.test_pid, {:notify_background, state.data.interactions})
    end
  end

  # ── Planner Agent ───────────────────────────────────────

  defmodule PlannerAgent do
    use Terra.Agent

    def init(args) do
      data = %{
        test_pid: args[:test_pid],
        queued_turns: [],
        current_messages: []
      }

      {:ok, :idle, data,
       provider: {PlannerProvider, %{}},
       kernel: args[:kernel]}
    end

    def context(_, state) do
      ctx =
        Terra.Context.new()
        |> Terra.Context.system("Build an action plan from: #{inspect(state.data.current_messages)}")
        |> Terra.Context.model(%{model: "mock"})
        |> Terra.Context.build()

      {ctx, state}
    end

    # Idle — start planning immediately
    def handle_input(:idle, {:process_turn, messages}, state) do
      state = %{state | data: %{state.data | current_messages: messages}}
      notify(state, {:plan, :planning})
      {:next_state, :planning, state, [:invoke]}
    end

    # Busy — queue the turn
    def handle_input(:planning, {:process_turn, messages}, state) do
      queue = state.data.queued_turns ++ [messages]
      state = %{state | data: %{state.data | queued_turns: queue}}
      notify(state, {:plan, :queued, length(queue)})
      {:keep_state, state}
    end

    def handle_input(:cooldown, {:process_turn, messages}, state) do
      queue = state.data.queued_turns ++ [messages]
      state = %{state | data: %{state.data | queued_turns: queue}}
      notify(state, {:plan, :queued, length(queue)})
      {:keep_state, state}
    end

    def handle_stream_event(_, _, state), do: {:keep_state, state}

    def handle_response(response, state) do
      # Extract text from response and publish to kernel
      text =
        response.content_blocks
        |> Enum.filter(&match?(%{type: "text"}, &1))
        |> Enum.map(& &1.text)
        |> Enum.join()

      doc = Document.new("plan", "Generated action plan", text)

      Kernel.lock(state.kernel, :plan)
      Kernel.write_and_unlock(state.kernel, :plan, doc)

      notify(state, {:plan, :published, text})

      # Cooldown before draining queue
      {:next_state, :cooldown, state, [{:state_timeout, 50, :drain}]}
    end

    def handle_timeout(:cooldown, :drain, state) do
      case state.data.queued_turns do
        [] ->
          notify(state, {:plan, :idle})
          {:next_state, :idle, state}

        turns ->
          # Take the latest queued turn (all accumulated messages)
          latest = List.last(turns)
          state = %{state | data: %{state.data | queued_turns: [], current_messages: latest}}
          notify(state, {:plan, :draining, length(turns)})
          {:next_state, :planning, state, [:invoke]}
      end
    end

    defp notify(state, event), do: send(state.data.test_pid, {:agent, event})
  end

  # ── Analysis Agent ──────────────────────────────────────

  defmodule AnalysisAgent do
    use Terra.Agent

    def init(args) do
      data = %{
        test_pid: args[:test_pid],
        queued_turns: [],
        current_messages: []
      }

      {:ok, :idle, data,
       provider: {AnalysisProvider, %{}},
       kernel: args[:kernel]}
    end

    def context(_, state) do
      # Read current plan buffer for analysis context
      plan = Kernel.read(state.kernel, :plan)

      ctx =
        Terra.Context.new()
        |> Terra.Context.system("Analyze conversation patterns")
        |> Terra.Context.document(plan)
        |> Terra.Context.history(state.data.current_messages)
        |> Terra.Context.model(%{model: "mock"})
        |> Terra.Context.age_tools(state)
        |> Terra.Context.build()

      {ctx, state}
    end

    def handle_input(:idle, {:analyze, messages}, state) do
      state = %{state | data: %{state.data | current_messages: messages}}
      notify(state, {:analysis, :analyzing})
      {:next_state, :analyzing, state, [:invoke]}
    end

    def handle_input(:analyzing, {:analyze, messages}, state) do
      queue = state.data.queued_turns ++ [messages]
      state = %{state | data: %{state.data | queued_turns: queue}}
      notify(state, {:analysis, :queued, length(queue)})
      {:keep_state, state}
    end

    def handle_input(:cooldown, {:analyze, messages}, state) do
      queue = state.data.queued_turns ++ [messages]
      state = %{state | data: %{state.data | queued_turns: queue}}
      notify(state, {:analysis, :queued, length(queue)})
      {:keep_state, state}
    end

    def handle_stream_event(_, _, state), do: {:keep_state, state}

    def handle_response(response, state) do
      text =
        response.content_blocks
        |> Enum.filter(&match?(%{type: "text"}, &1))
        |> Enum.map(& &1.text)
        |> Enum.join()

      doc = Document.new("analysis", "Conversation analysis", text)

      Kernel.lock(state.kernel, :analysis)
      Kernel.write_and_unlock(state.kernel, :analysis, doc)

      notify(state, {:analysis, :published, text})
      {:next_state, :cooldown, state, [{:state_timeout, 50, :drain}]}
    end

    def handle_timeout(:cooldown, :drain, state) do
      case state.data.queued_turns do
        [] ->
          notify(state, {:analysis, :idle})
          {:next_state, :idle, state}

        turns ->
          latest = List.last(turns)
          state = %{state | data: %{state.data | queued_turns: [], current_messages: latest}}
          notify(state, {:analysis, :draining, length(turns)})
          {:next_state, :analyzing, state, [:invoke]}
      end
    end

    defp notify(state, event), do: send(state.data.test_pid, {:agent, event})
  end

  # ── Tests ───────────────────────────────────────────────

  describe "full session lifecycle" do
    test "conversation + planner + analysis: end-to-end flow" do
      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:plan, :analysis],
          agents: [
            {ConvAgent, %{test_pid: self()}},
            {PlannerAgent, %{test_pid: self()}},
            {AnalysisAgent, %{test_pid: self()}}
          ]
        )

      [conv, planner, analysis] = Terra.Session.agents(sup)
      kernel = Terra.Session.kernel(sup)

      # Subscribe conversation to kernel updates
      Kernel.subscribe(kernel)

      # ── Turn 1: User sends message ──
      Terra.Agent.send_input(conv, {:message, "What should I do this weekend?"})

      assert_receive {:agent, {:conv, :processing}}, 500
      assert_receive {:agent, {:conv, :response, response_text}}, 500
      assert is_binary(response_text)

      # Orchestrator notifies background agents
      assert_receive {:notify_background, interactions}, 500

      # Trigger planner and analysis
      Terra.Agent.send_input(planner, {:process_turn, interactions})
      Terra.Agent.send_input(analysis, {:analyze, interactions})

      assert_receive {:agent, {:plan, :planning}}, 500
      assert_receive {:agent, {:analysis, :analyzing}}, 500

      # Planner publishes to kernel
      assert_receive {:agent, {:plan, :published, plan_text}}, 500
      assert plan_text =~ "Plan"

      # Kernel update notification
      assert_receive {:kernel_update, :plan, _doc}, 500

      # Analysis publishes to kernel
      assert_receive {:agent, {:analysis, :published, analysis_text}}, 500
      assert analysis_text =~ "activity recommendations"
      assert_receive {:kernel_update, :analysis, _doc}, 500

      # Verify kernel state
      snapshot = Kernel.snapshot(kernel)
      assert snapshot.plan != nil
      assert snapshot.analysis != nil
      assert snapshot.plan.content =~ "Plan"
      assert snapshot.analysis.content =~ "activity recommendations"

      # ── Turn 2: User sends another message ──
      # Kernel buffers should now be injected into conversation context
      Terra.Agent.send_input(conv, {:message, "What about indoor options if it rains?"})

      assert_receive {:agent, {:conv, :response, _}}, 500
      assert_receive {:notify_background, _interactions_2}, 500

      # Verify conversation context includes kernel documents
      {_, conv_state} = :sys.get_state(conv)
      {ctx, _state} = ConvAgent.context(:idle, conv_state)

      doc_titles = Enum.map(ctx.documents, & &1.title)
      assert "plan" in doc_titles
      assert "analysis" in doc_titles

      # Messages should include injected kernel documents
      [first_msg | _] = ctx.messages
      assert length(injected_documents(first_msg.content)) == 2
    end
  end

  describe "queue and cooldown" do
    test "planner agent queues turns while planning, drains after cooldown" do
      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:plan],
          agents: [
            {PlannerAgent, %{test_pid: self()}}
          ]
        )

      [planner] = Terra.Session.agents(sup)

      # Send first turn — starts planning
      Terra.Agent.send_input(planner, {:process_turn, [%{role: "user", content: "turn 1"}]})
      assert_receive {:agent, {:plan, :planning}}, 500

      # Send more turns while busy — should be queued
      Terra.Agent.send_input(planner, {:process_turn, [%{role: "user", content: "turn 2"}]})
      assert_receive {:agent, {:plan, :queued, 1}}, 500

      Terra.Agent.send_input(planner, {:process_turn, [%{role: "user", content: "turn 3"}]})
      assert_receive {:agent, {:plan, :queued, 2}}, 500

      # First plan completes → publish → cooldown
      assert_receive {:agent, {:plan, :published, _}}, 500

      # After cooldown, drains queue (processes latest)
      assert_receive {:agent, {:plan, :draining, 2}}, 500

      # Second plan runs and publishes
      assert_receive {:agent, {:plan, :published, _}}, 500

      # No more queued → idle
      assert_receive {:agent, {:plan, :idle}}, 500
    end

    test "analysis agent queues turns during cooldown" do
      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:analysis],
          agents: [
            {AnalysisAgent, %{test_pid: self()}}
          ]
        )

      [analysis] = Terra.Session.agents(sup)

      # First analysis
      Terra.Agent.send_input(analysis, {:analyze, [%{role: "user", content: "msg 1"}]})
      assert_receive {:agent, {:analysis, :analyzing}}, 500
      assert_receive {:agent, {:analysis, :published, _}}, 500

      # Queue during cooldown
      Terra.Agent.send_input(analysis, {:analyze, [%{role: "user", content: "msg 2"}]})
      assert_receive {:agent, {:analysis, :queued, 1}}, 500

      # Cooldown drains
      assert_receive {:agent, {:analysis, :draining, 1}}, 500
      assert_receive {:agent, {:analysis, :published, _}}, 500
      assert_receive {:agent, {:analysis, :idle}}, 500
    end
  end

  describe "kernel locking" do
    test "planner and analysis don't conflict via locking" do
      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:plan, :analysis],
          agents: [
            {PlannerAgent, %{test_pid: self()}},
            {AnalysisAgent, %{test_pid: self()}}
          ]
        )

      [planner, analysis] = Terra.Session.agents(sup)
      kernel = Terra.Session.kernel(sup)

      messages = [%{role: "user", content: "test"}]

      # Both start simultaneously — different slots, no conflict
      Terra.Agent.send_input(planner, {:process_turn, messages})
      Terra.Agent.send_input(analysis, {:analyze, messages})

      assert_receive {:agent, {:plan, :published, _}}, 500
      assert_receive {:agent, {:analysis, :published, _}}, 500

      # Both slots populated
      snapshot = Kernel.snapshot(kernel)
      assert snapshot.plan != nil
      assert snapshot.analysis != nil
    end
  end

  describe "supervisor restart" do
    test "kernel crash restarts all agents, agents reconnect" do
      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:plan],
          agents: [
            {ConvAgent, %{test_pid: self()}},
            {PlannerAgent, %{test_pid: self()}}
          ]
        )

      old_kernel = Terra.Session.kernel(sup)
      old_agents = Terra.Session.agents(sup)

      # Write something to kernel
      doc = Document.new("plan", "Facts", "old data")
      Kernel.write(old_kernel, :plan, doc)

      # Kill kernel — rest_for_one restarts everything after it
      Process.exit(old_kernel, :kill)
      Process.sleep(50)

      # New kernel and agents
      new_kernel = Terra.Session.kernel(sup)
      new_agents = Terra.Session.agents(sup)

      assert new_kernel != old_kernel
      assert new_agents != old_agents
      assert Process.alive?(new_kernel)
      assert Enum.all?(new_agents, &Process.alive?/1)

      # New kernel starts fresh (slots are nil)
      assert Kernel.read(new_kernel, :plan) == nil

      # New agents have the new kernel pid
      [conv, _planner] = new_agents
      {_, conv_state} = :sys.get_state(conv)
      assert conv_state.kernel == new_kernel
    end
  end

  describe "context builds with kernel documents" do
    test "conversation context includes kernel buffers as injected documents" do
      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:plan, :analysis],
          agents: [
            {ConvAgent, %{test_pid: self()}}
          ]
        )

      kernel = Terra.Session.kernel(sup)
      [conv] = Terra.Session.agents(sup)

      # Pre-populate kernel
      Kernel.write(kernel, :plan, Document.new("plan", "Facts", "User prefers outdoor activities"))
      Kernel.write(kernel, :analysis, Document.new("analysis", "Insights", "Outdoor activity preference detected"))

      # Trigger conversation
      Terra.Agent.send_input(conv, {:message, "What should I plan for tomorrow?"})
      assert_receive {:agent, {:conv, :response, _}}, 500

      # Inspect context that was built
      {_, state} = :sys.get_state(conv)
      {ctx, _state} = ConvAgent.context(:idle, state)

      assert ctx.system =~ "planning assistant"
      assert length(ctx.documents) == 2

      # Messages should have documents injected
      [first | _] = ctx.messages
      doc_contents =
        first.content
        |> injected_documents()
        |> Enum.map(& &1.content)

      assert Enum.any?(doc_contents, &(&1 =~ "User prefers outdoor activities"))
      assert Enum.any?(doc_contents, &(&1 =~ "Outdoor activity preference detected"))
    end

    test "nil kernel slots are skipped in context" do
      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:plan, :analysis],
          agents: [
            {ConvAgent, %{test_pid: self()}}
          ]
        )

      kernel = Terra.Session.kernel(sup)
      [conv] = Terra.Session.agents(sup)

      # Only populate plan, leave analysis nil
      Kernel.write(kernel, :plan, Document.new("plan", "Plan", "1. Plan a walk\n2. Pack water"))

      Terra.Agent.send_input(conv, {:message, "What's the plan?"})
      assert_receive {:agent, {:conv, :response, _}}, 500

      {_, state} = :sys.get_state(conv)
      {ctx, _state} = ConvAgent.context(:idle, state)

      # Only 1 document — analysis was nil, skipped
      assert length(ctx.documents) == 1
      assert hd(ctx.documents).title == "plan"
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
