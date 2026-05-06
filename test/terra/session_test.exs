defmodule Terra.SessionTest do
  use ExUnit.Case, async: true

  alias Terra.Kernel
  alias Terra.Document

  # ── Test Agents ─────────────────────────────────────────

  defmodule SimpleProvider do
    @behaviour Terra.Provider

    @impl true
    def stream(caller, _params) do
      ref = make_ref()

      spawn_link(fn ->
        for e <- [
              {:message_start, %{id: "msg_1", content: [], usage: %{input_tokens: 10, output_tokens: 0}}},
              {:content_block_start, 0, %{type: "text", text: ""}},
              {:content_block_delta, 0, %{type: "text_delta", text: "Hello"}},
              {:content_block_stop, 0},
              {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 5}},
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

  defmodule ConversationAgent do
    use Terra.Agent

    def init(args) do
      {:ok, :idle, %{test_pid: args[:test_pid]},
       provider: {SimpleProvider, %{}},
       kernel: args[:kernel]}
    end

    def context(_, state) do
      # Read from kernel — nil-safe
      plan = Kernel.read(state.kernel, :plan)

      ctx =
        Terra.Context.new()
        |> Terra.Context.system("You are helpful")
        |> Terra.Context.document(plan)
        |> Terra.Context.model(%{model: "mock"})
        |> Terra.Context.build()

      {ctx, state}
    end

    def handle_input(:idle, :go, state) do
      {:next_state, :processing, state, [:invoke]}
    end

    def handle_stream_event(_, _, state), do: {:keep_state, state}

    def handle_response(_response, state) do
      send(state.data.test_pid, {:conversation, :done})
      {:next_state, :idle, state}
    end
  end

  defmodule PlannerAgent do
    use Terra.Agent

    def init(args) do
      {:ok, :idle, %{test_pid: args[:test_pid]},
       provider: {SimpleProvider, %{}},
       kernel: args[:kernel]}
    end

    def context(_, state) do
      ctx =
        Terra.Context.new()
        |> Terra.Context.system("You are a planner. Produce a numbered action plan for the user's request.")
        |> Terra.Context.model(%{model: "mock"})
        |> Terra.Context.build()

      {ctx, state}
    end

    def handle_input(:idle, {:plan, _prompt}, state) do
      {:next_state, :planning, state, [:invoke]}
    end

    def handle_stream_event(_, _, state), do: {:keep_state, state}

    def handle_response(_response, state) do
      # Write the plan to kernel
      doc =
        Document.new(
          "plan",
          "Active plan",
          "1. Identify user location\n2. Fetch current conditions\n3. Recommend hydration if hot\n4. Suggest layers if cold"
        )

      Kernel.write(state.kernel, :plan, doc)
      send(state.data.test_pid, {:plan, :published})
      {:next_state, :idle, state}
    end
  end

  # ── Session Tests ───────────────────────────────────────

  describe "session supervisor" do
    test "starts kernel and multiple agents" do
      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:plan, :analysis],
          agents: [
            {ConversationAgent, %{test_pid: self()}},
            {PlannerAgent, %{test_pid: self()}}
          ]
        )

      assert Process.alive?(sup)

      # Kernel should be running
      kernel = Terra.Session.kernel(sup)
      assert Process.alive?(kernel)

      # Agents should be running
      agents = Terra.Session.agents(sup)
      assert length(agents) == 2
      assert Enum.all?(agents, &Process.alive?/1)
    end

    test "kernel ref is injected into each agent's state" do
      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:plan],
          agents: [
            {ConversationAgent, %{test_pid: self()}}
          ]
        )

      kernel = Terra.Session.kernel(sup)
      [agent_pid] = Terra.Session.agents(sup)

      {_state_name, state} = :sys.get_state(agent_pid)
      assert state.kernel == kernel
    end

    test "agents can read/write kernel" do
      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:plan],
          agents: [
            {ConversationAgent, %{test_pid: self()}},
            {PlannerAgent, %{test_pid: self()}}
          ]
        )

      kernel = Terra.Session.kernel(sup)
      [_conversation, planner] = Terra.Session.agents(sup)

      # Planner agent builds a plan and writes to kernel
      Terra.Agent.send_input(planner, {:plan, "user said something"})
      assert_receive {:plan, :published}, 500

      # Kernel should have the document (wrapped in buffer tags)
      doc = Kernel.read(kernel, :plan)
      assert doc.content =~ "Recommend hydration if hot"
    end

    test "conversation agent reads kernel documents in context" do
      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:plan],
          agents: [
            {ConversationAgent, %{test_pid: self()}},
            {PlannerAgent, %{test_pid: self()}}
          ]
        )

      [conversation, planner] = Terra.Session.agents(sup)

      # First: planner agent publishes
      Terra.Agent.send_input(planner, {:plan, "data"})
      assert_receive {:plan, :published}, 500

      # Then: conversation reads kernel in context
      Terra.Agent.send_input(conversation, :go)
      assert_receive {:conversation, :done}, 500

      # Verify conversation's context included the kernel doc
      {_, conv_state} = :sys.get_state(conversation)
      {ctx, _state} = ConversationAgent.context(:idle, conv_state)

      assert length(ctx.documents) == 1
      assert hd(ctx.documents).content =~ "Recommend hydration if hot"
    end

    test "rest_for_one: kernel crash restarts agents" do
      {:ok, sup} =
        Terra.Session.start_link(
          buffers: [:plan],
          agents: [
            {ConversationAgent, %{test_pid: self()}}
          ]
        )

      kernel = Terra.Session.kernel(sup)
      [agent] = Terra.Session.agents(sup)

      # Kill kernel
      Process.exit(kernel, :kill)
      Process.sleep(50)

      # New kernel and agent should be up
      new_kernel = Terra.Session.kernel(sup)
      [new_agent] = Terra.Session.agents(sup)

      assert new_kernel != kernel
      assert new_agent != agent
      assert Process.alive?(new_kernel)
      assert Process.alive?(new_agent)
    end
  end
end
