defmodule Terra.Tool do
  @moduledoc """
  Pipeline builder for tool definitions.

  Eliminates the boilerplate of manually constructing nested JSON Schema maps
  for `Terra.ToolRegistry` `tools/1`. Each function returns the accumulator,
  so definitions read as a top-down pipeline.

  ## Example

      import Terra.Tool

      tool("get_weather")
      |> desc("Get current weather for a city")
      |> param(:city, :string, required: true, desc: "City name")
      |> aging(expiry: 4)
      |> expiry_message(~S"<%= @input[:city] %> weather data expired — re-fetch if needed.")
      |> build()

  Produces the same map you'd write by hand:

      %{
        name: "get_weather",
        description: "Get current weather for a city",
        input_schema: %{
          type: "object",
          properties: %{city: %{type: "string", description: "City name"}},
          required: ["city"]
        },
        expiry_distance: 4,
        expiry_message: "<%= @input[:city] %> weather data expired — re-fetch if needed."
      }

  ## Nested Objects

      param(:location, :object, required: true, desc: "Location", properties: [
        param(:latitude, :number, required: true, desc: "Latitude in decimal degrees"),
        param(:longitude, :number, required: true, desc: "Longitude in decimal degrees")
      ])

  ## Enums

      param(:condition, :string, required: true, enum: ~w(sunny cloudy rainy snowy))

  ## Array Parameters

      param(:conditions, :array, required: true,
        desc: "Conditions to query",
        items: %{type: "string", enum: ~w(sunny cloudy rainy)})

  Or with the shorthand:

      param(:conditions, {:array, :string}, required: true,
        desc: "Conditions to query",
        enum: ~w(sunny cloudy rainy))
  """

  defstruct [
    :name,
    :description,
    :expiry_distance,
    :pruning_distance,
    :result_template,
    :expiry_message,
    :tail_hints,
    :result_type,
    :cache_control,
    params: []
  ]

  @type param_entry :: {atom(), map(), boolean()}
  @type t :: %__MODULE__{}
  @type acc :: t()

  @doc """
  Start a new tool definition with the given name.
  """
  @spec tool(String.t()) :: t()
  def tool(name) when is_binary(name) do
    %__MODULE__{name: name}
  end

  @doc """
  Set the tool description.
  """
  @spec desc(t(), String.t()) :: t()
  def desc(%__MODULE__{} = acc, description) do
    %{acc | description: description}
  end

  @doc """
  Add a parameter to the tool's input schema.

  ## Options

    * `:required` — whether the parameter is required (default: `false`)
    * `:desc` — parameter description
    * `:enum` — list of allowed values
    * `:items` — item schema for array types (map)
    * `:properties` — nested param list for object types (from `param/3` calls)
    * `:default` — default value
    * `:minimum` / `:maximum` — numeric constraints

  ## Type Shorthands

    * `{:array, :string}` — shorthand for array of strings (enum applied to items)
    * `{:array, :integer}` — shorthand for array of integers
  """
  @spec param(t(), atom(), atom() | {atom(), atom()}, keyword()) :: t()

  def param(%__MODULE__{} = acc, name, {:array, item_type}, opts) do
    {required?, opts} = Keyword.pop(opts, :required, false)
    {desc_val, opts} = Keyword.pop(opts, :desc)
    {enum_val, opts} = Keyword.pop(opts, :enum)

    items = %{type: to_string(item_type)}
    items = if enum_val, do: Map.put(items, :enum, enum_val), else: items

    schema = %{type: "array", items: items}
    schema = if desc_val, do: Map.put(schema, :description, desc_val), else: schema
    schema = merge_constraints(schema, opts)

    %{acc | params: acc.params ++ [{name, schema, required?}]}
  end

  def param(%__MODULE__{} = acc, name, type, opts) when is_atom(type) do
    {required?, opts} = Keyword.pop(opts, :required, false)
    {desc_val, opts} = Keyword.pop(opts, :desc)
    {enum_val, opts} = Keyword.pop(opts, :enum)
    {items_val, opts} = Keyword.pop(opts, :items)
    {properties, opts} = Keyword.pop(opts, :properties)

    schema = %{type: to_string(type)}
    schema = if desc_val, do: Map.put(schema, :description, desc_val), else: schema
    schema = if enum_val, do: Map.put(schema, :enum, enum_val), else: schema
    schema = if items_val, do: Map.put(schema, :items, items_val), else: schema
    schema = merge_constraints(schema, opts)

    schema =
      if type == :object && is_list(properties) do
        {nested_props, nested_required} = compile_params(properties)
        schema = Map.put(schema, :properties, nested_props)
        if nested_required != [], do: Map.put(schema, :required, Enum.map(nested_required, &to_string/1)), else: schema
      else
        schema
      end

    %{acc | params: acc.params ++ [{name, schema, required?}]}
  end

  @doc """
  Build a standalone parameter (for use inside `:properties` lists of nested objects).

  Returns `{name, schema}` instead of updating an accumulator.

  ## Example

      param(:latitude, :number, required: true, desc: "Latitude in decimal degrees")
  """
  @spec param(atom(), atom() | {atom(), atom()}, keyword()) :: {atom(), map(), boolean()}
  def param(name, type, opts) when is_atom(name) do
    {required?, opts} = Keyword.pop(opts, :required, false)
    {desc_val, opts} = Keyword.pop(opts, :desc)
    {enum_val, opts} = Keyword.pop(opts, :enum)
    {items_val, opts} = Keyword.pop(opts, :items)
    {properties, opts} = Keyword.pop(opts, :properties)

    schema =
      case type do
        {:array, item_type} ->
          items = %{type: to_string(item_type)}
          items = if enum_val, do: Map.put(items, :enum, enum_val), else: items
          %{type: "array", items: items}

        _ ->
          s = %{type: to_string(type)}
          s = if enum_val, do: Map.put(s, :enum, enum_val), else: s
          s = if items_val, do: Map.put(s, :items, items_val), else: s
          s
      end

    schema = if desc_val, do: Map.put(schema, :description, desc_val), else: schema
    schema = merge_constraints(schema, opts)

    schema =
      if type == :object && is_list(properties) do
        {nested_props, nested_required} = compile_params(properties)
        schema = Map.put(schema, :properties, nested_props)
        if nested_required != [], do: Map.put(schema, :required, Enum.map(nested_required, &to_string/1)), else: schema
      else
        schema
      end

    {name, schema, required?}
  end

  @doc """
  Set context aging distances.

  ## Options

    * `:expiry` — `expiry_distance` (assistant turns before result expires)
    * `:pruning` — `pruning_distance` (turns before result is pruned entirely)
  """
  @spec aging(t(), keyword()) :: t()
  def aging(%__MODULE__{} = acc, opts) do
    acc
    |> maybe_put(:expiry_distance, Keyword.get(opts, :expiry))
    |> maybe_put(:pruning_distance, Keyword.get(opts, :pruning))
  end

  @doc """
  Set the EEx template for rendering active results.
  """
  @spec result_template(t(), String.t()) :: t()
  def result_template(%__MODULE__{} = acc, template) do
    %{acc | result_template: template}
  end

  @doc """
  Set the EEx template shown when the result has expired.
  """
  @spec expiry_message(t(), String.t()) :: t()
  def expiry_message(%__MODULE__{} = acc, message) do
    %{acc | expiry_message: message}
  end

  @doc """
  Set hints appended after the tool result in the LLM context.

  Tail hints are additional instructions injected right after the tool's
  result content. Use them to steer the LLM's interpretation of the result
  (e.g., "Focus on the warm-cold front interaction" or "Summarize in 2 sentences").
  """
  @spec tail_hints(t(), String.t()) :: t()
  def tail_hints(%__MODULE__{} = acc, hints) do
    %{acc | tail_hints: hints}
  end

  @doc """
  Set the result type (`:text` or `:document`).

  - `:text` (default) — the result is serialized as a plain `tool_result`
    content block.
  - `:document` — the result is wrapped in a `Terra.Document` struct and
    serialized as a document block inside the tool result. Use this for
    large, stable outputs (forecasts, profiles) that benefit from
    provider-level prompt caching.
  """
  @spec result_type(t(), :text | :document) :: t()
  def result_type(%__MODULE__{} = acc, type) when type in [:text, :document] do
    %{acc | result_type: type}
  end

  @doc """
  Set cache control for Anthropic prompt caching.

  When set, the tool definition itself is marked as cacheable in the API
  request. This is useful for tools with expensive definitions (large
  descriptions or many parameters) that don't change between invocations —
  Anthropic will cache the tool definition to reduce input token costs.

  Passing `:ephemeral` is a shorthand for `%{type: "ephemeral"}`.
  """
  @spec cache_control(t(), :ephemeral | map()) :: t()
  def cache_control(%__MODULE__{} = acc, :ephemeral), do: %{acc | cache_control: %{type: "ephemeral"}}
  def cache_control(%__MODULE__{} = acc, value) when is_map(value), do: %{acc | cache_control: value}

  @doc """
  Finalize the accumulator into a tool definition map.

  Compiles the collected params into `input_schema` and strips
  internal keys (`_params`, `_required`).
  """
  @spec build(t()) :: Terra.ToolRegistry.tool()
  def build(%__MODULE__{} = tool) do
    {properties, _nested_req} = compile_params(tool.params)

    required =
      tool.params
      |> Enum.filter(fn {_name, _schema, req?} -> req? end)
      |> Enum.map(fn {name, _schema, _req?} -> to_string(name) end)

    input_schema = %{type: "object", properties: properties}

    input_schema =
      if required != [] do
        Map.put(input_schema, :required, required)
      else
        input_schema
      end

    result = %{name: tool.name, input_schema: input_schema}

    result
    |> maybe_put(:description, tool.description)
    |> maybe_put(:expiry_distance, tool.expiry_distance)
    |> maybe_put(:pruning_distance, tool.pruning_distance)
    |> maybe_put(:result_template, tool.result_template)
    |> maybe_put(:expiry_message, tool.expiry_message)
    |> maybe_put(:tail_hints, tool.tail_hints)
    |> maybe_put(:result_type, tool.result_type)
    |> maybe_put(:cache_control, tool.cache_control)
  end

  # ── Parse ───────────────────────────────────────────────

  @doc """
  Reconstruct a `%Terra.Tool{}` from a tool definition map (reverse of `build/1`).

  Walks `input_schema.properties` and `input_schema.required` to rebuild
  the typed params list.
  """
  @spec parse(map()) :: t()
  def parse(tool_def) when is_map(tool_def) do
    input_schema = Map.get(tool_def, :input_schema, %{})
    properties = Map.get(input_schema, :properties, %{})
    required = Map.get(input_schema, :required, [])

    params = parse_properties(properties, required)

    %__MODULE__{
      name: tool_def[:name],
      description: tool_def[:description],
      expiry_distance: tool_def[:expiry_distance],
      pruning_distance: tool_def[:pruning_distance],
      result_template: tool_def[:result_template],
      expiry_message: tool_def[:expiry_message],
      tail_hints: tool_def[:tail_hints],
      result_type: tool_def[:result_type],
      cache_control: tool_def[:cache_control],
      params: params
    }
  end

  defp parse_properties(properties, required) do
    Enum.map(properties, fn {name, schema} ->
      name = if is_binary(name), do: String.to_atom(name), else: name
      required? = to_string(name) in required
      {name, schema, required?}
    end)
  end

  # ── Validate ────────────────────────────────────────────

  @doc """
  Validate raw input (JSON string or decoded map) against the tool schema.

  Returns `{:ok, validated_map}` with atom keys, coerced types, and defaults
  applied, or `{:error, [error_string]}`.
  """
  @spec validate(t(), String.t() | map()) :: {:ok, map()} | {:error, [String.t()]}
  def validate(%__MODULE__{} = tool, input) when is_binary(input) do
    case String.trim(input) do
      "" ->
        validate(tool, %{})

      trimmed ->
        case Jason.decode(trimmed) do
          {:ok, decoded} -> validate(tool, decoded)
          {:error, _} -> {:error, ["input: invalid JSON"]}
        end
    end
  end

  def validate(%__MODULE__{} = tool, input) when is_map(input) do
    validate_params(tool.params, input, "")
  end

  defp validate_params(params, input, prefix) do
    {result, errors} =
      Enum.reduce(params, {%{}, []}, fn {name, schema, required?}, {acc, errs} ->
        key = to_string(name)
        path = if prefix == "", do: to_string(name), else: "#{prefix}.#{name}"

        case Map.fetch(input, key) do
          {:ok, value} ->
            case validate_value(value, schema, path) do
              {:ok, coerced} -> {Map.put(acc, name, coerced), errs}
              {:error, new_errs} -> {acc, errs ++ new_errs}
            end

          :error ->
            cond do
              required? ->
                {acc, errs ++ ["#{path}: is required"]}

              Map.has_key?(schema, :default) ->
                {Map.put(acc, name, schema.default), errs}

              true ->
                {acc, errs}
            end
        end
      end)

    if errors == [], do: {:ok, result}, else: {:error, errors}
  end

  defp validate_value(value, schema, path) do
    with {:ok, coerced} <- coerce_type(value, schema, path),
         :ok <- check_enum(coerced, schema, path),
         :ok <- check_range(coerced, schema, path) do
      {:ok, coerced}
    end
  end

  # ── Type coercion ──

  defp coerce_type(value, %{type: "string"}, _path) when is_binary(value), do: {:ok, value}
  defp coerce_type(_value, %{type: "string"}, path), do: {:error, ["#{path}: must be a string"]}

  defp coerce_type(value, %{type: "integer"}, _path) when is_integer(value), do: {:ok, value}

  defp coerce_type(value, %{type: "integer"}, path) when is_binary(value) do
    case Integer.parse(value) do
      {n, ""} -> {:ok, n}
      _ -> {:error, ["#{path}: must be an integer"]}
    end
  end

  defp coerce_type(_value, %{type: "integer"}, path), do: {:error, ["#{path}: must be an integer"]}

  defp coerce_type(value, %{type: "number"}, _path) when is_number(value), do: {:ok, value}

  defp coerce_type(value, %{type: "number"}, path) when is_binary(value) do
    case Float.parse(value) do
      {n, ""} -> {:ok, n}
      _ -> {:error, ["#{path}: must be a number"]}
    end
  end

  defp coerce_type(_value, %{type: "number"}, path), do: {:error, ["#{path}: must be a number"]}

  defp coerce_type(true, %{type: "boolean"}, _path), do: {:ok, true}
  defp coerce_type(false, %{type: "boolean"}, _path), do: {:ok, false}
  defp coerce_type("true", %{type: "boolean"}, _path), do: {:ok, true}
  defp coerce_type("false", %{type: "boolean"}, _path), do: {:ok, false}
  defp coerce_type(_, %{type: "boolean"}, path), do: {:error, ["#{path}: must be a boolean"]}

  defp coerce_type(value, %{type: "object"} = schema, path) when is_map(value) do
    case Map.get(schema, :properties) do
      nil ->
        {:ok, value}

      properties ->
        nested_params = properties_to_params(properties, Map.get(schema, :required, []))
        validate_params(nested_params, value, path)
    end
  end

  defp coerce_type(_, %{type: "object"}, path), do: {:error, ["#{path}: must be an object"]}

  defp coerce_type(value, %{type: "array"} = schema, path) when is_list(value) do
    item_schema = Map.get(schema, :items, %{})

    value
    |> Enum.with_index()
    |> Enum.reduce({[], []}, fn {item, idx}, {acc, errs} ->
      item_path = "#{path}[#{idx}]"

      case validate_value(item, item_schema, item_path) do
        {:ok, coerced} -> {acc ++ [coerced], errs}
        {:error, new_errs} -> {acc, errs ++ new_errs}
      end
    end)
    |> case do
      {items, []} -> {:ok, items}
      {_, errors} -> {:error, errors}
    end
  end

  defp coerce_type(_, %{type: "array"}, path), do: {:error, ["#{path}: must be an array"]}

  # Fallback — pass through unknown types
  defp coerce_type(value, _schema, _path), do: {:ok, value}

  # ── Enum check ──

  defp check_enum(value, %{enum: allowed}, path) do
    if value in allowed, do: :ok, else: {:error, ["#{path}: must be one of: #{Enum.join(allowed, ", ")}"]}
  end

  defp check_enum(_value, _schema, _path), do: :ok

  # ── Range check ──

  defp check_range(value, schema, path) when is_number(value) do
    min_err =
      case Map.get(schema, :minimum) do
        nil -> nil
        min when value < min -> "#{path}: must be >= #{min}"
        _ -> nil
      end

    max_err =
      case Map.get(schema, :maximum) do
        nil -> nil
        max when value > max -> "#{path}: must be <= #{max}"
        _ -> nil
      end

    errors = Enum.reject([min_err, max_err], &is_nil/1)
    if errors == [], do: :ok, else: {:error, errors}
  end

  defp check_range(_value, _schema, _path), do: :ok

  # ── Helpers ──

  defp properties_to_params(properties, required_list) when is_map(properties) do
    Enum.map(properties, fn {name, schema} ->
      name = if is_binary(name), do: String.to_atom(name), else: name
      required? = to_string(name) in required_list
      {name, schema, required?}
    end)
  end

  # ── Private ──────────────────────────────────────────────

  defp compile_params(params) do
    Enum.reduce(params, {%{}, []}, fn
      {name, schema, true}, {props, req} ->
        {Map.put(props, name, schema), req ++ [name]}

      {name, schema, _required?}, {props, req} ->
        {Map.put(props, name, schema), req}
    end)
  end

  defp merge_constraints(schema, opts) do
    Enum.reduce(opts, schema, fn
      {:default, v}, s -> Map.put(s, :default, v)
      {:minimum, v}, s -> Map.put(s, :minimum, v)
      {:maximum, v}, s -> Map.put(s, :maximum, v)
      _, s -> s
    end)
  end

  defp maybe_put(acc, _key, nil), do: acc
  defp maybe_put(acc, key, value), do: Map.put(acc, key, value)
end
