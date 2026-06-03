defmodule Terra.MixProject do
  use Mix.Project

  def project do
    [
      app: :terra,
      version: "0.1.0",
      elixir: "~> 1.16",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      docs: docs(),
      package: package()
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  def application do
    [
      extra_applications: [:logger],
      mod: {Terra.Application, []}
    ]
  end

  defp deps do
    [
      {:jason, "~> 1.4"},
      {:req, "~> 0.5"},
      {:telemetry, "~> 1.0"},
      {:ex_doc, "~> 0.37", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => "https://github.com/anuvaya/terra"}
    ]
  end

  defp docs do
    [
      main: "Terra",
      source_url: "https://github.com/anuvaya/terra",
      source_ref: "main",
      nest_modules_by_prefix: [Terra.Provider],
      extras: [
        "guides/getting-started.md",
        "guides/cheatsheet.cheatmd"
      ],
      groups_for_extras: [
        Guides: ~r/guides\/.*/
      ],
      groups_for_modules: [
        "Core": [Terra.Agent, Terra.Agent.State, Terra.Context, Terra.Tool, Terra.ToolRegistry, Terra.Document],
        "Providers": [Terra.Provider, Terra.Provider.Anthropic, Terra.Provider.OpenAI, Terra.Provider.Google],
        "Multi-Agent": [Terra.Session, Terra.Kernel],
        "Internals": [Terra.SSE, Terra.APIError, Terra.Application]
      ]
    ]
  end
end
