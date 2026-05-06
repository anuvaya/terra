defmodule Terra.Application do
  @moduledoc false

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      {Registry, keys: :unique, name: Terra.Session.Registry}
    ]

    Supervisor.start_link(children, strategy: :one_for_one, name: Terra.Supervisor)
  end
end
