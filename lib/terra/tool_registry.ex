defmodule Terra.ToolRegistry do
  @moduledoc """
  Behaviour for defining tool registries.

  A registry groups related `Terra.Tool` definitions and implements their
  execution. Tools are eagerly executed when their `content_block_stop`
  arrives during streaming, so results are ready by the time
  `c:Terra.Agent.handle_response/2` is called.

  ## Example

      defmodule MyTools do
        use Terra.ToolRegistry

        import Terra.Tool

        @impl true
        def tools(_state) do
          [
            tool("get_weather")
            |> desc("Get current weather for a city")
            |> param(:city, :string, required: true, desc: "City name")
            |> aging(expiry: 4)
            |> expiry_message(~S"<%= @input[:city] %> weather data expired — re-fetch if needed.")
          ]
        end

        @impl true
        def execute("get_weather", %{city: city}, state) do
          {:ok, %{temp: 72, city: city, conditions: "sunny"}, state}
        end
      end

  `tools/1` receives the agent `%State{}` and returns `[%Terra.Tool{}]` structs
  — no `Terra.Tool.build/1` call needed. Terra calls `build/1` internally when
  serializing for the provider API. Registries that don't need state can ignore it.

  Register tools in your agent's `c:Terra.Agent.init/1`:

      def init(args) do
        {:ok, :idle, %{}, provider: {Terra.Provider.Anthropic, %{}}, registries: [MyTools]}
      end

  ## The `helpers` option

  Pass a module via `use Terra.ToolRegistry, helpers: MyHelpers` to make it
  available as `@h` in `result_template` EEx templates:

      defmodule MyTools do
        use Terra.ToolRegistry, helpers: MyHelpers

        import Terra.Tool

        @impl true
        def tools(_state) do
          [
            tool("get_forecast")
            |> result_template(~S"<%= @h.format_forecast(@result) %>")
            # ...
          ]
        end

        # ...
      end

  The helpers module is passed to `render_result/5` and bound to the `@h`
  assign during EEx evaluation. This is useful for sharing formatting logic
  across multiple tool templates.
  """

  @type tool :: map()

  @callback tools(state :: Terra.Agent.State.t()) :: [Terra.Tool.t()]

  @callback execute(name :: String.t(), input :: map(), state :: Terra.Agent.State.t()) ::
              {:ok, term(), Terra.Agent.State.t()} | {:error, term(), Terra.Agent.State.t()}

  @doc false
  def render_result(result, _input, nil, _state, _helpers), do: result

  def render_result(result, _input, %Terra.Tool{result_template: nil, result_type: nil}, _state, _helpers), do: result

  def render_result(result, input, %Terra.Tool{} = tool, state, helpers) do
    rendered =
      case tool.result_template do
        nil -> result
        template ->
          assigns = [result: result, input: input, state: state]
          assigns = if helpers, do: Keyword.put(assigns, :h, helpers), else: assigns
          EEx.eval_string(template, assigns: assigns)
      end

    case tool.result_type do
      :document ->
        Terra.Document.new(tool.name, tool.name, rendered)

      _ ->
        rendered
    end
  end

  defmacro __using__(opts \\ []) do
    helpers = Keyword.get(opts, :helpers)

    quote do
      @behaviour Terra.ToolRegistry

      @doc false
      def __has_tool__?(name, state), do: Enum.any?(tools(state), &(&1.name == name))

      @doc false
      def __tool_def__(name, state), do: Enum.find(tools(state), &(&1.name == name))

      @doc false
      def handle_execution(name, raw_input, %Terra.Agent.State{} = state) do
        tool = __tool_def__(name, state)

        case Terra.Tool.validate(tool, raw_input) do
          {:ok, input} ->
            case execute(name, input, state) do
              {:ok, raw_result, new_state} ->
                rendered = Terra.ToolRegistry.render_result(raw_result, input, tool, new_state, unquote(helpers))
                {:ok, rendered, new_state}

              {:error, reason, new_state} ->
                {:error, reason, new_state}
            end

          {:error, errors} ->
            {:error, "Invalid input: #{Enum.join(errors, "; ")}", state}
        end
      end
    end
  end
end
