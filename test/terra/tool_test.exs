defmodule Terra.ToolTest do
  use ExUnit.Case, async: true

  import Terra.Tool

  describe "basic tool" do
    test "builds a simple tool with one required param" do
      result =
        tool("get_weather")
        |> desc("Get current weather for a city")
        |> param(:city, :string, required: true, desc: "City name")
        |> aging(expiry: 4)
        |> expiry_message(~S"<%= @input[:city] %> weather data expired.")
        |> build()

      assert result.name == "get_weather"
      assert result.description == "Get current weather for a city"
      assert result.expiry_distance == 4
      assert result.expiry_message == ~S"<%= @input[:city] %> weather data expired."

      assert result.input_schema == %{
               type: "object",
               properties: %{
                 city: %{type: "string", description: "City name"}
               },
               required: ["city"]
             }
    end

    test "omits required list when no params are required" do
      result =
        tool("check_status")
        |> desc("Check status")
        |> param(:verbose, :boolean, desc: "Verbose output")
        |> build()

      refute Map.has_key?(result.input_schema, :required)
      assert result.input_schema.properties.verbose == %{type: "boolean", description: "Verbose output"}
    end
  end

  describe "nested objects" do
    test "builds nested object params with required fields" do
      result =
        tool("get_observation")
        |> desc("Get weather observation")
        |> param(:datetime, :string, required: true, desc: "ISO 8601 datetime")
        |> param(:location, :object, required: true,
             desc: "Location",
             properties: [
               param(:latitude, :number, required: true, desc: "Latitude"),
               param(:longitude, :number, required: true, desc: "Longitude")
             ])
        |> build()

      assert result.input_schema.required == ["datetime", "location"]

      location = result.input_schema.properties.location
      assert location.type == "object"
      assert location.description == "Location"
      assert location.required == ["latitude", "longitude"]
      assert location.properties.latitude == %{type: "number", description: "Latitude"}
      assert location.properties.longitude == %{type: "number", description: "Longitude"}
    end
  end

  describe "arrays" do
    test "array with items map" do
      result =
        tool("fetch_locations")
        |> desc("Fetch locations")
        |> param(:conditions, :array, required: true,
             desc: "Conditions",
             items: %{type: "string", enum: ~w(sunny cloudy rainy)})
        |> build()

      conditions = result.input_schema.properties.conditions
      assert conditions.type == "array"
      assert conditions.items == %{type: "string", enum: ~w(sunny cloudy rainy)}
    end

    test "array shorthand {:array, :string} with enum" do
      result =
        tool("fetch_locations")
        |> desc("Fetch locations")
        |> param(:conditions, {:array, :string}, required: true,
             desc: "Conditions",
             enum: ~w(sunny cloudy rainy))
        |> build()

      conditions = result.input_schema.properties.conditions
      assert conditions.type == "array"
      assert conditions.items == %{type: "string", enum: ~w(sunny cloudy rainy)}
    end
  end

  describe "aging" do
    test "sets expiry and pruning distances" do
      result =
        tool("test")
        |> desc("Test")
        |> aging(expiry: 2, pruning: 4)
        |> build()

      assert result.expiry_distance == 2
      assert result.pruning_distance == 4
    end

    test "sets only expiry" do
      result =
        tool("test")
        |> desc("Test")
        |> aging(expiry: 3)
        |> build()

      assert result.expiry_distance == 3
      refute Map.has_key?(result, :pruning_distance)
    end
  end

  describe "optional fields" do
    test "result_template, tail_hints, result_type, cache_control" do
      result =
        tool("test")
        |> desc("Test")
        |> result_template(~S"<%= @result %>")
        |> tail_hints("Check the data")
        |> result_type(:document)
        |> cache_control(:ephemeral)
        |> build()

      assert result.result_template == ~S"<%= @result %>"
      assert result.tail_hints == "Check the data"
      assert result.result_type == :document
      assert result.cache_control == %{type: "ephemeral"}
    end
  end

  describe "numeric constraints" do
    test "minimum, maximum, default" do
      result =
        tool("test")
        |> desc("Test")
        |> param(:depth, :integer, minimum: 1, maximum: 5, default: 2, desc: "Depth level")
        |> build()

      depth = result.input_schema.properties.depth
      assert depth.minimum == 1
      assert depth.maximum == 5
      assert depth.default == 2
    end
  end

  describe "no internal keys leak" do
    test "build strips _params and _required" do
      result =
        tool("test")
        |> desc("Test")
        |> build()

      refute Map.has_key?(result, :_params)
      refute Map.has_key?(result, :_required)
    end
  end

  # ── validate/2 ──────────────────────────────────────────────

  describe "validate/2 — required fields" do
    setup do
      schema =
        tool("test")
        |> param(:name, :string, required: true)
        |> param(:age, :integer, [])

      %{schema: schema}
    end

    test "passes with all required fields present", %{schema: schema} do
      assert {:ok, %{name: "Alice"}} = Terra.Tool.validate(schema, ~s({"name": "Alice"}))
    end

    test "fails when required field is missing", %{schema: schema} do
      assert {:error, errors} = Terra.Tool.validate(schema, ~s({"age": 25}))
      assert "name: is required" in errors
    end

    test "accepts JSON string input", %{schema: schema} do
      assert {:ok, %{name: "Bob"}} = Terra.Tool.validate(schema, ~s({"name": "Bob"}))
    end

    test "accepts already-decoded map input", %{schema: schema} do
      assert {:ok, %{name: "Bob"}} = Terra.Tool.validate(schema, %{"name" => "Bob"})
    end
  end

  describe "validate/2 — empty input" do
    test "treats empty string as no arguments" do
      schema = tool("noargs")
      assert {:ok, %{}} = Terra.Tool.validate(schema, "")
    end

    test "treats whitespace-only string as no arguments" do
      schema = tool("noargs")
      assert {:ok, %{}} = Terra.Tool.validate(schema, "   \n  ")
    end

    test "empty input still enforces required fields" do
      schema =
        tool("test")
        |> param(:name, :string, required: true)

      assert {:error, errors} = Terra.Tool.validate(schema, "")
      assert "name: is required" in errors
    end

    test "applies defaults when input is empty" do
      schema =
        tool("test")
        |> param(:depth, :integer, default: 5)

      assert {:ok, %{depth: 5}} = Terra.Tool.validate(schema, "")
    end
  end

  describe "validate/2 — type coercion" do
    test "coerces string to integer" do
      schema = tool("test") |> param(:depth, :integer, [])
      assert {:ok, %{depth: 3}} = Terra.Tool.validate(schema, ~s({"depth": "3"}))
    end

    test "coerces string to number (float)" do
      schema = tool("test") |> param(:lat, :number, [])
      assert {:ok, %{lat: 3.5}} = Terra.Tool.validate(schema, ~s({"lat": "3.5"}))
    end

    test "coerces string to boolean" do
      schema = tool("test") |> param(:verbose, :boolean, [])
      assert {:ok, %{verbose: true}} = Terra.Tool.validate(schema, ~s({"verbose": "true"}))
      assert {:ok, %{verbose: false}} = Terra.Tool.validate(schema, ~s({"verbose": "false"}))
    end

    test "passes through correct types unchanged" do
      schema = tool("test") |> param(:n, :integer, []) |> param(:s, :string, [])
      assert {:ok, %{n: 42, s: "hi"}} = Terra.Tool.validate(schema, ~s({"n": 42, "s": "hi"}))
    end

    test "rejects non-coercible values" do
      schema = tool("test") |> param(:n, :integer, [])
      assert {:error, errors} = Terra.Tool.validate(schema, ~s({"n": "abc"}))
      assert "n: must be an integer" in errors
    end
  end

  describe "validate/2 — defaults" do
    test "applies default when field absent" do
      schema = tool("test") |> param(:depth, :integer, default: 2)
      assert {:ok, %{depth: 2}} = Terra.Tool.validate(schema, ~s({}))
    end

    test "provided value overrides default" do
      schema = tool("test") |> param(:depth, :integer, default: 2)
      assert {:ok, %{depth: 5}} = Terra.Tool.validate(schema, ~s({"depth": 5}))
    end
  end

  describe "validate/2 — enum" do
    test "passes when value in enum" do
      schema = tool("test") |> param(:unit, :string, enum: ~w(celsius fahrenheit))
      assert {:ok, %{unit: "celsius"}} = Terra.Tool.validate(schema, ~s({"unit": "celsius"}))
    end

    test "fails when value not in enum" do
      schema = tool("test") |> param(:unit, :string, enum: ~w(celsius fahrenheit))
      assert {:error, errors} = Terra.Tool.validate(schema, ~s({"unit": "kelvin"}))
      assert "unit: must be one of: celsius, fahrenheit" in errors
    end
  end

  describe "validate/2 — range" do
    test "passes within range" do
      schema = tool("test") |> param(:depth, :integer, minimum: 1, maximum: 5)
      assert {:ok, %{depth: 3}} = Terra.Tool.validate(schema, ~s({"depth": 3}))
    end

    test "fails below minimum" do
      schema = tool("test") |> param(:depth, :integer, minimum: 1, maximum: 5)
      assert {:error, errors} = Terra.Tool.validate(schema, ~s({"depth": 0}))
      assert "depth: must be >= 1" in errors
    end

    test "fails above maximum" do
      schema = tool("test") |> param(:depth, :integer, minimum: 1, maximum: 5)
      assert {:error, errors} = Terra.Tool.validate(schema, ~s({"depth": 10}))
      assert "depth: must be <= 5" in errors
    end
  end

  describe "validate/2 — nested objects" do
    test "validates nested required fields" do
      schema =
        tool("test")
        |> param(:location, :object, required: true,
             properties: [
               param(:latitude, :number, required: true),
               param(:longitude, :number, required: true)
             ])

      assert {:error, errors} =
               Terra.Tool.validate(schema, ~s({"location": {"latitude": 28.6}}))

      assert "location.longitude: is required" in errors
    end

    test "passes with valid nested object" do
      schema =
        tool("test")
        |> param(:location, :object, required: true,
             properties: [
               param(:latitude, :number, required: true),
               param(:longitude, :number, required: true)
             ])

      assert {:ok, %{location: %{latitude: 28.6, longitude: 77.2}}} =
               Terra.Tool.validate(schema, ~s({"location": {"latitude": 28.6, "longitude": 77.2}}))
    end
  end

  describe "validate/2 — arrays" do
    test "validates array items against enum" do
      schema =
        tool("test")
        |> param(:conditions, {:array, :string}, enum: ~w(sunny cloudy rainy))

      assert {:ok, %{conditions: ["sunny", "cloudy"]}} =
               Terra.Tool.validate(schema, ~s({"conditions": ["sunny", "cloudy"]}))
    end

    test "fails on invalid array item" do
      schema =
        tool("test")
        |> param(:conditions, {:array, :string}, enum: ~w(sunny cloudy rainy))

      assert {:error, errors} =
               Terra.Tool.validate(schema, ~s({"conditions": ["sunny", "stormy"]}))

      assert "conditions[1]: must be one of: sunny, cloudy, rainy" in errors
    end
  end

  describe "validate/2 — unknown keys" do
    test "silently drops unknown keys" do
      schema = tool("test") |> param(:name, :string, required: true)

      assert {:ok, %{name: "Alice"}} =
               Terra.Tool.validate(schema, ~s({"name": "Alice", "extra": "junk"}))

      assert {:ok, result} = Terra.Tool.validate(schema, ~s({"name": "Alice", "extra": "junk"}))
      refute Map.has_key?(result, :extra)
    end
  end

  # ── parse/1 ─────────────────────────────────────────────

  describe "parse/1 — reconstruct from definition map" do
    test "parses a simple tool definition" do
      tool_def = %{
        name: "get_weather",
        description: "Get weather",
        input_schema: %{
          type: "object",
          properties: %{city: %{type: "string"}},
          required: ["city"]
        },
        expiry_distance: 4
      }

      schema = Terra.Tool.parse(tool_def)
      assert %Terra.Tool{name: "get_weather"} = schema

      # Can validate with the parsed schema
      assert {:ok, %{city: "Mumbai"}} = Terra.Tool.validate(schema, ~s({"city": "Mumbai"}))
      assert {:error, _} = Terra.Tool.validate(schema, ~s({}))
    end

    test "parses nested objects" do
      tool_def = %{
        name: "get_observation",
        input_schema: %{
          type: "object",
          properties: %{
            location: %{
              type: "object",
              properties: %{
                latitude: %{type: "number"},
                longitude: %{type: "number"}
              },
              required: ["latitude", "longitude"]
            }
          },
          required: ["location"]
        }
      }

      schema = Terra.Tool.parse(tool_def)

      assert {:error, errors} =
               Terra.Tool.validate(schema, ~s({"location": {"latitude": 28.6}}))

      assert "location.longitude: is required" in errors
    end

    test "roundtrip: build → parse → build produces same output" do
      original =
        tool("test")
        |> desc("Test tool")
        |> param(:x, :integer, required: true, minimum: 1, maximum: 10)
        |> param(:y, :string, enum: ~w(a b c))
        |> aging(expiry: 4, pruning: 8)

      built = build(original)
      roundtripped = original |> build() |> Terra.Tool.parse() |> build()
      assert roundtripped == built
    end
  end
end
