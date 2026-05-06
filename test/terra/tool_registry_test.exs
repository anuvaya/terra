defmodule Terra.ToolRegistryTest do
  use ExUnit.Case, async: true

  defmodule TestRegistry do
    use Terra.ToolRegistry

    @impl true
    def tools(_state) do
      [
        %{
          name: "fetch_forecast",
          description: "Fetch weather forecast with selective sections",
          input_schema: %{
            type: "object",
            properties: %{
              forecast_id: %{type: "string"},
              sections: %{type: "array", items: %{enum: ~w(temperature precipitation wind)}}
            },
            required: ["forecast_id"]
          },
          expiry_distance: 4,
          pruning_distance: :infinity,
          result_template: ~S"""
          === Weather Forecast ===
          <%= @result["name"] %>
          """,
          expiry_message: ~S"""
          [Forecast data expired. Re-fetch if needed.]
          """,
          tail_hints: ~S"""
          Remember to interpret readings against meteorological standards.
          """
        },
        %{
          name: "get_alerts",
          description: "Calculate weather alerts",
          input_schema: %{
            type: "object",
            properties: %{
              date: %{type: "string"}
            },
            required: ["date"]
          }
        }
      ]
    end

    @impl true
    def execute("fetch_forecast", input, state) do
      {:ok, %{"name" => "Test Forecast", "forecast_id" => input[:forecast_id]}, state}
    end

    def execute("get_alerts", input, state) do
      {:ok, %{"date" => input[:date], "alerts" => []}, state}
    end

    def execute(_, _, state), do: {:error, :unknown_tool, state}
  end

  describe "tools/1" do
    test "returns list of tool definitions" do
      tools = TestRegistry.tools(%Terra.Agent.State{})
      assert length(tools) == 2
    end

    test "each tool has name, description, and input_schema" do
      [forecast, alerts] = TestRegistry.tools(%Terra.Agent.State{})

      assert forecast.name == "fetch_forecast"
      assert forecast.description == "Fetch weather forecast with selective sections"
      assert forecast.input_schema.type == "object"
      assert forecast.input_schema.required == ["forecast_id"]

      assert alerts.name == "get_alerts"
    end

    test "tools can have optional Terra-specific fields" do
      [forecast, alerts] = TestRegistry.tools(%Terra.Agent.State{})

      assert forecast.expiry_distance == 4
      assert forecast.pruning_distance == :infinity
      assert forecast.result_template =~ "Weather Forecast"
      assert forecast.expiry_message =~ "expired"
      assert forecast.tail_hints =~ "meteorological standards"

      # Optional fields can be omitted
      refute Map.has_key?(alerts, :expiry_distance)
      refute Map.has_key?(alerts, :result_template)
      refute Map.has_key?(alerts, :tail_hints)
    end
  end

  describe "execute/3" do
    test "dispatches by tool name and returns {:ok, result, state}" do
      assert {:ok, %{"name" => "Test Forecast"}, %{}} =
               TestRegistry.execute("fetch_forecast", %{forecast_id: "abc"}, %{})
    end

    test "passes input to the handler" do
      assert {:ok, %{"date" => "2026-01-01"}, %{}} =
               TestRegistry.execute("get_alerts", %{date: "2026-01-01"}, %{})
    end

    test "returns {:error, reason, state} for unknown tools" do
      assert {:error, :unknown_tool, %{}} = TestRegistry.execute("nope", %{}, %{})
    end
  end
end
