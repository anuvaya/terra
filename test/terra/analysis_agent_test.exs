defmodule Terra.AnalysisAgentTest do
  @moduledoc """
  Isolated test for the analysis agent pattern: bootstrap → analyze →
  tool writes to kernel → cooldown → drain queue → idle.

  Uses a mock provider that emits update_analysis tool_use events and
  the real Terra.Kernel for locking/unlocking verification.
  """
  use ExUnit.Case, async: true

  alias Terra.Kernel
  alias Terra.Document

  # ── Mock Provider ─────────────────────────────────────

  defmodule AnalysisProvider do
    @behaviour Terra.Provider

    @impl true
    def stream(caller, params) do
      ref = make_ref()
      table = params[:_test_table]

      call_count =
        if table do
          :ets.update_counter(table, :calls, 1)
        else
          1
        end

      spawn_link(fn ->
        events =
          case table && :ets.lookup(table, {:events, call_count}) do
            [{_, evts}] -> evts
            _ -> default_tool_events(call_count)
          end

        for e <- events, do: send(caller, {:stream, ref, e})
      end)

      {:ok, ref}
    end

    @impl true
    def cancel(_ref), do: :ok

    defp default_tool_events(n) do
      tool_id = "toolu_analysis_#{n}"
      content = "Foundation analysis round #{n}. Heavy rain in Seattle indicates flooding risk."

      [
        {:message_start, %{id: "msg_#{n}", content: [], usage: %{input_tokens: 50, output_tokens: 0}}},
        {:content_block_start, 0, %{type: "tool_use", id: tool_id, name: "update_analysis"}},
        {:content_block_delta, 0, %{type: "input_json_delta", partial_json: Jason.encode!(%{content: content})}},
        {:content_block_stop, 0},
        {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 20}},
        :message_stop
      ]
    end
  end

  # ── Tool Registry ────────────────────────────────────

  defmodule AnalysisTools do
    use Terra.ToolRegistry
    import Terra.Tool

    @impl true
    def tools(_state) do
      [
        tool("update_analysis")
        |> desc("Write updated analysis to buffer")
        |> param(:content, :string, required: true),
        tool("continue_analysis")
        |> desc("Request another round")
        |> param(:focus, :string, required: true)
      ]
    end

    @impl true
    def execute("update_analysis", %{content: content}, state) do
      if String.trim(content) != "" do
        doc = Document.new("analysis", "Deep weather analysis", content)
        Kernel.write_and_unlock(state.kernel, :analysis, doc)
      end

      {:ok, "Buffer updated.", state}
    end

    def execute("continue_analysis", %{focus: focus}, state) do
      {:ok, "Continuing. Focus: #{focus}.", state}
    end
  end

  # ── Analysis Agent ───────────────────────────────────

  defmodule TestAnalysisAgent do
    use Terra.Agent

    def trigger_analysis(pid, focus \\ "general", cancel_existing \\ false) do
      Terra.Agent.send_input(pid, {:analyze, focus, cancel_existing})
    end

    @impl true
    def init(args) do
      data = %{
        test_pid: args[:test_pid],
        current_focus: "foundation",
        queue: [],
        provider_config: args[:provider_config] || %{}
      }

      {:ok, :idle, data,
       provider: {AnalysisProvider, data.provider_config},
       kernel: args[:kernel],
       registries: [AnalysisTools],
       actions: [{:state_timeout, 0, :bootstrap}]}
    end

    @impl true
    def context(_, state) do
      d = state.data
      existing = Kernel.read(state.kernel, :analysis)

      ctx =
        Terra.Context.new()
        |> Terra.Context.system("You are a weather analysis engine.")
        |> Terra.Context.document(existing)
        |> Terra.Context.history([%{role: "user", content: "Focus: #{d.current_focus}"}])
        |> Terra.Context.model(%{
          model: "mock",
          _test_table: d.provider_config[:table]
        })
        |> Terra.Context.build()

      {ctx, state}
    end

    @impl true
    def handle_input(:idle, {:analyze, focus, _cancel}, state) do
      Kernel.lock(state.kernel, :analysis)
      notify(state, {:analysis, :triggered, focus})
      {:next_state, :analyzing, %{state | data: %{state.data | current_focus: focus}}, [:invoke]}
    end

    def handle_input(:analyzing, {:analyze, focus, true}, state) do
      Kernel.unlock(state.kernel, :analysis)
      Kernel.lock(state.kernel, :analysis)
      notify(state, {:analysis, :cancelled_and_restarted, focus})
      {:keep_state, %{state | data: %{state.data | current_focus: focus}}, [:cancel_stream, :invoke]}
    end

    def handle_input(:analyzing, {:analyze, focus, _cancel}, state) do
      queue = state.data.queue ++ [focus]
      notify(state, {:analysis, :queued, focus, length(queue)})
      {:keep_state, %{state | data: %{state.data | queue: queue}}}
    end

    def handle_input(:cooldown, {:analyze, focus, _cancel}, state) do
      queue = state.data.queue ++ [focus]
      notify(state, {:analysis, :queued, focus, length(queue)})
      {:keep_state, %{state | data: %{state.data | queue: queue}}}
    end

    def handle_input(_, _, state), do: {:keep_state, state}

    @impl true
    def handle_response(_response, state) do
      Kernel.unlock(state.kernel, :analysis)
      notify(state, {:analysis, :complete, state.data.current_focus})
      {:next_state, :cooldown, state, [{:state_timeout, 50, :drain}]}
    end

    @impl true
    def handle_error(_state_name, error, state) do
      Kernel.unlock(state.kernel, :analysis)
      notify(state, {:analysis, :error, error})
      {:next_state, :cooldown, state, [{:state_timeout, 50, :drain}]}
    end

    @impl true
    def handle_timeout(:idle, :bootstrap, state) do
      Kernel.lock(state.kernel, :analysis)
      notify(state, {:analysis, :bootstrap})
      {:next_state, :analyzing, state, [:invoke]}
    end

    def handle_timeout(:cooldown, :drain, state) do
      case state.data.queue do
        [next_focus | rest] ->
          Kernel.lock(state.kernel, :analysis)
          d = %{state.data | queue: rest, current_focus: next_focus}
          notify(state, {:analysis, :draining, next_focus})
          {:next_state, :analyzing, %{state | data: d}, [:invoke]}

        [] ->
          notify(state, {:analysis, :idle})
          {:next_state, :idle, state}
      end
    end

    def handle_timeout(_, _, state), do: {:keep_state, state}

    defp notify(state, event), do: send(state.data.test_pid, {:agent, event})
  end

  # ── Tests ────────────────────────────────────────────

  describe "bootstrap" do
    test "agent starts idle, then immediately bootstraps into analysis" do
      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:analysis],
          agents: [{TestAnalysisAgent, %{test_pid: self()}}]
        )

      # Bootstrap fires via state_timeout 0
      assert_receive {:agent, {:analysis, :bootstrap}}, 500
      assert_receive {:agent, {:analysis, :complete, "foundation"}}, 500
      assert_receive {:agent, {:analysis, :idle}}, 500

      # Kernel should have the analysis buffer
      kernel = Terra.Session.kernel(sup)
      doc = Kernel.read(kernel, :analysis)
      assert doc != nil
      assert doc.content =~ "<analysis-buffer"
      assert doc.content =~ "Heavy rain in Seattle"
      assert doc.content =~ "state=\"unlocked\""
    end
  end

  describe "trigger from conversation agent" do
    test "trigger while idle locks kernel, analyzes, unlocks" do
      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:analysis],
          agents: [{TestAnalysisAgent, %{test_pid: self()}}]
        )

      [agent] = Terra.Session.agents(sup)
      kernel = Terra.Session.kernel(sup)

      # Wait for bootstrap to finish
      assert_receive {:agent, {:analysis, :bootstrap}}, 500
      assert_receive {:agent, {:analysis, :idle}}, 500

      # Trigger new analysis
      TestAnalysisAgent.trigger_analysis(agent, "temperature")
      assert_receive {:agent, {:analysis, :triggered, "temperature"}}, 500
      assert_receive {:agent, {:analysis, :complete, "temperature"}}, 500
      assert_receive {:agent, {:analysis, :idle}}, 500

      # Kernel unlocked after completion
      assert :ok = Kernel.lock(kernel, :analysis)
      Kernel.unlock(kernel, :analysis)
    end
  end

  describe "queuing" do
    test "requests while analyzing are queued and drained" do
      table = :ets.new(:analysis_queue_test, [:public, :set])
      :ets.insert(table, {:calls, 0})

      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:analysis],
          agents: [{TestAnalysisAgent, %{test_pid: self(), provider_config: %{table: table}}}]
        )

      [agent] = Terra.Session.agents(sup)

      # Wait for bootstrap
      assert_receive {:agent, {:analysis, :bootstrap}}, 500
      assert_receive {:agent, {:analysis, :idle}}, 500

      # Trigger analysis, then queue more while busy
      TestAnalysisAgent.trigger_analysis(agent, "temperature")
      assert_receive {:agent, {:analysis, :triggered, "temperature"}}, 500

      # These arrive while analyzing
      TestAnalysisAgent.trigger_analysis(agent, "humidity")
      assert_receive {:agent, {:analysis, :queued, "humidity", 1}}, 500

      TestAnalysisAgent.trigger_analysis(agent, "wind")
      assert_receive {:agent, {:analysis, :queued, "wind", 2}}, 500

      # First completes → cooldown → drain
      assert_receive {:agent, {:analysis, :complete, "temperature"}}, 500
      assert_receive {:agent, {:analysis, :draining, "humidity"}}, 500
      assert_receive {:agent, {:analysis, :complete, "humidity"}}, 500
      assert_receive {:agent, {:analysis, :draining, "wind"}}, 500
      assert_receive {:agent, {:analysis, :complete, "wind"}}, 500
      assert_receive {:agent, {:analysis, :idle}}, 500
    end

    test "requests during cooldown are queued" do
      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:analysis],
          agents: [{TestAnalysisAgent, %{test_pid: self()}}]
        )

      [agent] = Terra.Session.agents(sup)

      # Wait for bootstrap to complete (enters cooldown briefly)
      assert_receive {:agent, {:analysis, :bootstrap}}, 500
      assert_receive {:agent, {:analysis, :complete, "foundation"}}, 500

      # Send during cooldown window (before :idle)
      TestAnalysisAgent.trigger_analysis(agent, "precipitation")
      assert_receive {:agent, {:analysis, :queued, "precipitation", 1}}, 500

      # Cooldown drains the queue
      assert_receive {:agent, {:analysis, :draining, "precipitation"}}, 500
      assert_receive {:agent, {:analysis, :complete, "precipitation"}}, 500
      assert_receive {:agent, {:analysis, :idle}}, 500
    end
  end

  describe "cancel while analyzing" do
    test "cancel_existing=true restarts analysis with new focus" do
      table = :ets.new(:analysis_cancel_test, [:public, :set])
      :ets.insert(table, {:calls, 0})

      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:analysis],
          agents: [{TestAnalysisAgent, %{test_pid: self(), provider_config: %{table: table}}}]
        )

      [agent] = Terra.Session.agents(sup)

      # Wait for bootstrap
      assert_receive {:agent, {:analysis, :bootstrap}}, 500
      assert_receive {:agent, {:analysis, :idle}}, 500

      # Start analysis
      TestAnalysisAgent.trigger_analysis(agent, "temperature")
      assert_receive {:agent, {:analysis, :triggered, "temperature"}}, 500

      # Cancel and restart with new focus
      TestAnalysisAgent.trigger_analysis(agent, "urgent_storm", true)
      assert_receive {:agent, {:analysis, :cancelled_and_restarted, "urgent_storm"}}, 500
      assert_receive {:agent, {:analysis, :complete, "urgent_storm"}}, 500
    end
  end

  describe "kernel buffer format" do
    test "read returns XML-wrapped content with state and timestamp" do
      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:analysis],
          agents: [{TestAnalysisAgent, %{test_pid: self()}}]
        )

      kernel = Terra.Session.kernel(sup)

      # Wait for bootstrap to write buffer
      assert_receive {:agent, {:analysis, :bootstrap}}, 500
      assert_receive {:agent, {:analysis, :complete, "foundation"}}, 500

      doc = Kernel.read(kernel, :analysis)
      assert doc.title == "analysis"
      assert doc.context == "Deep weather analysis"
      assert doc.content =~ ~r/<analysis-buffer state="unlocked" updated-at="\d{4}-\d{2}-\d{2}T/
      assert doc.content =~ "</analysis-buffer>"
    end

    test "read shows locked state during analysis" do
      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:analysis],
          agents: [{TestAnalysisAgent, %{test_pid: self()}}]
        )

      kernel = Terra.Session.kernel(sup)

      # Wait for bootstrap to complete, then trigger new analysis
      assert_receive {:agent, {:analysis, :bootstrap}}, 500
      assert_receive {:agent, {:analysis, :idle}}, 500

      # Trigger and immediately read while analyzing
      [agent] = Terra.Session.agents(sup)
      TestAnalysisAgent.trigger_analysis(agent, "temperature")
      assert_receive {:agent, {:analysis, :triggered, "temperature"}}, 500

      # The buffer has content from bootstrap, and the slot is now locked
      doc = Kernel.read(kernel, :analysis)
      assert doc.content =~ "state=\"locked\""

      # Wait for completion
      assert_receive {:agent, {:analysis, :complete, "temperature"}}, 500
    end

    test "nil buffer before first write" do
      {:ok, kernel} = Kernel.start_link(buffers: [:analysis])
      assert Kernel.read(kernel, :analysis) == nil
    end
  end

  describe "context building" do
    test "existing analysis buffer is injected as document in context" do
      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:analysis],
          agents: [{TestAnalysisAgent, %{test_pid: self()}}]
        )

      [agent] = Terra.Session.agents(sup)

      # Wait for bootstrap to populate kernel
      assert_receive {:agent, {:analysis, :bootstrap}}, 500
      assert_receive {:agent, {:analysis, :idle}}, 500

      # Inspect the agent's context
      {_, state} = :sys.get_state(agent)
      {ctx, _state} = TestAnalysisAgent.context(:idle, state)

      # Should have the analysis document
      assert length(ctx.documents) == 1
      [doc] = ctx.documents
      assert doc.title == "analysis"
      assert doc.content =~ "analysis-buffer"
      assert doc.content =~ "Heavy rain in Seattle"
    end
  end
end
