defmodule Terra.Session do
  @moduledoc """
  Supervisor for multi-agent sessions.

  Starts a shared `Terra.Kernel` and N agents under a `rest_for_one`
  supervisor. If the kernel crashes, all agents restart. The kernel
  pid is automatically injected into each agent's init args.

  ## Usage

      Terra.Session.start_link(
        buffers: [:plan, :analysis],
        agents: [
          {ConversationAgent, %{user_id: "123"}},
          {PlannerAgent, %{}},
          {AnalysisAgent, %{}}
        ]
      )

  Agents can also pass `start_link` opts (e.g. `:name` for process registration):

      Terra.Session.start_link(
        buffers: [:plan],
        agents: [
          {ConversationAgent, %{id: "123"}, name: {:via, Registry, {MyRegistry, "123"}}}
        ]
      )
  """

  use Supervisor

  def start_link(opts) do
    {name, opts} = Keyword.pop(opts, :name)

    if name do
      Supervisor.start_link(__MODULE__, opts, name: name)
    else
      Supervisor.start_link(__MODULE__, opts)
    end
  end

  @doc """
  Return the `Terra.Kernel` pid for this session.
  """
  @spec kernel(Supervisor.supervisor()) :: pid()
  def kernel(sup) do
    sup
    |> Supervisor.which_children()
    |> Enum.find_value(fn
      {:kernel, pid, _, _} -> pid
      _ -> nil
    end)
  end

  @doc """
  Send input to all agents in the session.
  """
  @spec send_input(Supervisor.supervisor(), term()) :: :ok
  def send_input(sup, input) do
    for pid <- agents(sup), do: Terra.Agent.send_input(pid, input)
    :ok
  end

  @doc """
  Return the agent pids for this session, in the order they were specified
  in the `:agents` option.
  """
  @spec agents(Supervisor.supervisor()) :: [pid()]
  def agents(sup) do
    sup
    |> Supervisor.which_children()
    |> Enum.filter(fn
      {{:agent, _}, _pid, _, _} -> true
      _ -> false
    end)
    |> Enum.sort_by(fn {{:agent, idx}, _, _, _} -> idx end)
    |> Enum.map(fn {_, pid, _, _} -> pid end)
  end

  @impl true
  def init(opts) do
    buffers = Keyword.fetch!(opts, :buffers)
    agent_specs = Keyword.get(opts, :agents, [])

    # Generate a unique registry key for this session's kernel
    kernel_key = {__MODULE__, make_ref()}

    kernel_child = %{
      id: :kernel,
      start: {Terra.Kernel, :start_link, [[buffers: buffers, name: {:via, Registry, {Terra.Session.Registry, kernel_key}}]]},
      type: :worker,
      restart: :permanent,
      significant: false
    }

    agent_children =
      Enum.with_index(agent_specs, fn spec, idx ->
        {module, args, agent_opts} = normalize_agent_spec(spec)

        %{
          id: {:agent, idx},
          start: {__MODULE__, :start_agent, [module, args, agent_opts, kernel_key]},
          type: :worker,
          restart: :transient,
          significant: true
        }
      end)

    children = [kernel_child | agent_children]
    Supervisor.init(children, strategy: :rest_for_one, auto_shutdown: :any_significant)
  end

  @doc false
  def start_agent(module, args, opts, kernel_key) do
    [{kernel_pid, _}] = Registry.lookup(Terra.Session.Registry, kernel_key)
    args = Map.put(args, :kernel, kernel_pid)
    module.start_link(args, opts)
  end

  defp normalize_agent_spec({module, args, opts}) when is_list(opts), do: {module, args, opts}
  defp normalize_agent_spec({module, args}), do: {module, args, []}
end
