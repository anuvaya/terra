defmodule Terra.Context do
  @moduledoc """
  Builder for assembling the LLM context window.

  Called every invoke via the consumer's `context/2` callback. The pipeline
  sets up system prompt, documents, history, and model config, then applies
  optional transforms (tool aging, thinking pruning) before `build/1`
  assembles the final message list.

  ## Usage

      def context(_state_name, state) do
        Terra.Context.new()
        |> Terra.Context.system("You are a weather forecaster")
        |> Terra.Context.document(forecast_doc)
        |> Terra.Context.history(state.data.interactions)
        |> Terra.Context.model(%{model: "claude-sonnet-4-5-20250929", max_tokens: 2048})
        |> Terra.Context.age_tools(state)
        |> Terra.Context.prune_thinking(1)
        |> Terra.Context.build()
      end

  ## Tool Aging

  `age_tools/2` ages tool results in history based on distance (assistant
  turns from the end) and per-tool configuration from registries on the
  agent state:

  | Status    | Condition                                     | What LLM Sees        |
  |-----------|-----------------------------------------------|----------------------|
  | Active    | distance < expiry_distance                    | `result_template`    |
  | Expired   | expiry_distance ≤ distance < pruning_distance | `expiry_message`     |
  | Pruned    | distance ≥ pruning_distance                   | Omitted              |

  No `expiry_distance` → stays active forever.
  No `pruning_distance` → expired results stay as expiry_message forever.

  ## Thinking Pruning

  `prune_thinking/2` strips `thinking` blocks from assistant messages older
  than a given distance. Thinking blocks are large and only useful for
  recent reasoning — the text output already captures the conclusion.
  """

  @type t :: %__MODULE__{
          system: String.t() | nil,
          documents: [Terra.Document.t()],
          history: [map()],
          messages: [map()],
          model: map()
        }

  defstruct system: nil,
            documents: [],
            history: [],
            messages: [],
            model: %{}

  @doc """
  Create an empty context builder.
  """
  @spec new() :: t()
  def new, do: %__MODULE__{}

  @doc """
  Set the system prompt.
  """
  @spec system(t(), String.t()) :: t()
  def system(%__MODULE__{} = ctx, prompt) do
    %{ctx | system: prompt}
  end

  @doc """
  Add a `Terra.Document` to the context.

  Documents are injected as structured blocks at the start of the message
  list, enabling prompt caching for large stable content.

  Passing `nil` or `{:error, _}` is a no-op — this lets you safely pipe
  results from `Terra.Kernel.read/2` without checking for errors:

      ctx
      |> Terra.Context.document(Terra.Kernel.read(kernel, :forecast))
  """
  @spec document(t(), Terra.Document.t() | nil) :: t()
  def document(%__MODULE__{} = ctx, nil), do: ctx
  def document(%__MODULE__{} = ctx, {:error, _}), do: ctx

  def document(%__MODULE__{} = ctx, %Terra.Document{} = doc) do
    %{ctx | documents: ctx.documents ++ [doc]}
  end

  @doc """
  Set the conversation history (list of message maps with `:role` and `:content`).
  """
  @spec history(t(), [map()]) :: t()
  def history(%__MODULE__{} = ctx, messages) do
    %{ctx | history: messages}
  end

  @doc """
  Set the model configuration (`:model`, `:max_tokens`, `:config`, and any
  provider-specific fields like `:temperature` or `:thinking`).
  """
  @spec model(t(), map()) :: t()
  def model(%__MODULE__{} = ctx, config) do
    %{ctx | model: config}
  end

  # ── Transforms ──────────────────────────────────────────

  @doc """
  Age tool results in history based on per-tool expiry and pruning distances.

  Reads tool definitions from `state.registries` to determine aging behaviour
  for each tool result. Tool results are aged based on their distance
  (number of assistant turns) from the end of the conversation.

  This is a pipeline step — call it before `build/1`:

      Terra.Context.new()
      |> Terra.Context.history(state.data.history)
      |> Terra.Context.age_tools(state)
      |> Terra.Context.build()
  """
  @spec age_tools(t(), Terra.Agent.State.t()) :: t()
  def age_tools(%__MODULE__{} = ctx, %Terra.Agent.State{registries: []}), do: ctx

  def age_tools(%__MODULE__{} = ctx, %Terra.Agent.State{} = state) do
    %{ctx | history: do_age_history(ctx.history, state)}
  end

  @doc """
  Strip `thinking` blocks from assistant messages older than `distance` turns.

  Thinking blocks are large and expensive context. After a few turns, the
  text output already captures the conclusion — keeping the thinking chain
  just wastes tokens. A distance of `1` keeps thinking only on the most
  recent assistant message.

      Terra.Context.new()
      |> Terra.Context.history(state.data.history)
      |> Terra.Context.prune_thinking(1)
      |> Terra.Context.build()
  """
  @spec prune_thinking(t(), non_neg_integer()) :: t()
  def prune_thinking(%__MODULE__{} = ctx, distance) when is_integer(distance) and distance >= 0 do
    %{ctx | history: do_prune_thinking(ctx.history, distance)}
  end

  # ── Build ───────────────────────────────────────────────

  @doc """
  Assemble the final message list.

  Injects document blocks at the start of the first user message and
  populates `ctx.messages` from the (possibly transformed) history.
  """
  @spec build(t()) :: t()
  def build(%__MODULE__{documents: [], history: []} = ctx) do
    %{ctx | messages: []}
  end

  def build(%__MODULE__{} = ctx) do
    prefix_blocks = build_document_blocks(ctx.documents)
    %{ctx | messages: inject_prefix(prefix_blocks, ctx.history)}
  end

  defp build_document_blocks(documents), do: documents

  defp inject_prefix([], history), do: history

  defp inject_prefix(prefix_blocks, []) do
    [%{role: "user", content: prefix_blocks}]
  end

  defp inject_prefix(prefix_blocks, [%{role: "user"} = first | rest]) do
    first_content = normalize_content(first.content)
    [%{first | content: prefix_blocks ++ first_content} | rest]
  end

  defp inject_prefix(prefix_blocks, history) do
    [%{role: "user", content: prefix_blocks} | history]
  end

  defp normalize_content(content) when is_binary(content) do
    [%{type: "text", text: content}]
  end

  defp normalize_content(content) when is_list(content), do: content

  # ── Thinking Pruning ───────────────────────────────────────

  defp do_prune_thinking(history, distance) do
    distances = compute_distances(history)

    history
    |> Enum.zip(distances)
    |> Enum.map(fn {msg, dist} ->
      if msg.role == "assistant" and is_list(msg.content) and dist >= distance do
        content = Enum.reject(msg.content, &match?(%{type: "thinking"}, &1))

        case content do
          [] -> msg
          _ -> %{msg | content: content}
        end
      else
        msg
      end
    end)
  end

  # ── Tool Aging ─────────────────────────────────────────────

  defp do_age_history(history, state) do
    tool_defs = collect_tool_defs(state)
    distances = compute_distances(history)
    tool_use_index = build_tool_use_index(history)

    history
    |> Enum.zip(distances)
    |> Enum.flat_map(fn {msg, distance} ->
      age_message(msg, distance, tool_defs, tool_use_index)
    end)
  end

  defp collect_tool_defs(%Terra.Agent.State{registries: registries} = state) do
    registries
    |> Enum.flat_map(& &1.tools(state))
    |> Map.new(fn tool -> {tool.name, tool} end)
  end

  # Distance = number of assistant messages AFTER this position.
  defp compute_distances(history) do
    history
    |> Enum.reverse()
    |> Enum.map_reduce(0, fn msg, count ->
      if msg.role == "assistant" do
        {count, count + 1}
      else
        {count, count}
      end
    end)
    |> elem(0)
    |> Enum.reverse()
  end

  defp has_tool_use?(content) do
    Enum.any?(content, &match?(%{type: "tool_use"}, &1))
  end

  defp has_tool_result?(content) do
    Enum.any?(content, &match?(%{type: "tool_result"}, &1))
  end

  # Build a map of tool_use_id → {tool_name, input} from history
  defp build_tool_use_index(history) do
    Enum.reduce(history, %{}, fn msg, acc ->
      case msg do
        %{role: "assistant", content: content} when is_list(content) ->
          Enum.reduce(content, acc, fn
            %{type: "tool_use", id: id, name: name, input: input}, acc ->
              Map.put(acc, id, %{name: name, input: input})

            %{type: "tool_use", id: id, name: name}, acc ->
              Map.put(acc, id, %{name: name, input: %{}})

            _, acc ->
              acc
          end)

        _ ->
          acc
      end
    end)
  end

  defp age_message(msg, distance, tool_defs, tool_use_index) do
    cond do
      msg.role == "assistant" and is_list(msg.content) and has_tool_use?(msg.content) ->
        age_assistant_tool_message(msg, distance, tool_defs)

      msg.role == "user" and is_list(msg.content) and has_tool_result?(msg.content) ->
        age_tool_result_message(msg, distance, tool_defs, tool_use_index)

      true ->
        [msg]
    end
  end

  # A thinking-only assistant message is rejected by Anthropic with
  # `messages.N.content.M: thinking blocks cannot be modified`.
  # If filtering left only thinking blocks, drop
  # the entire message — its matching tool_result user message will be pruned
  # by `age_tool_result_message` in the same pass, so we don't leave orphans.
  defp age_assistant_tool_message(msg, distance, tool_defs) do
    aged_content =
      Enum.filter(msg.content, fn
        %{type: "tool_use", name: name} ->
          case Map.get(tool_defs, name) do
            nil -> true
            tool_def -> not pruned?(distance, tool_def)
          end

        _ ->
          true
      end)

    cond do
      aged_content == [] -> []
      thinking_only?(aged_content) -> []
      true -> [%{msg | content: aged_content}]
    end
  end

  defp thinking_only?(content) do
    Enum.all?(content, fn block ->
      type = Map.get(block, :type) || Map.get(block, "type")
      type in ["thinking", "redacted_thinking"]
    end)
  end

  defp age_tool_result_message(msg, distance, tool_defs, tool_use_index) do
    {aged_content, tail_hints} =
      msg.content
      |> Enum.reduce({[], []}, fn
        %{type: "tool_result", tool_use_id: id} = result, {content, hints} ->
          case Map.get(tool_use_index, id) do
            nil ->
              {content ++ [result], hints}

            %{name: name, input: input} ->
              tool_def = Map.get(tool_defs, name)
              {aged, new_hints} = age_single_tool_result(result, distance, tool_def, input)
              {content ++ aged, hints ++ new_hints}
          end

        other, {content, hints} ->
          {content ++ [other], hints}
      end)

    # Append tail_hints AFTER all tool_results to avoid breaking the API
    final_content = aged_content ++ tail_hints

    case final_content do
      [] -> []
      content -> [%{msg | content: content}]
    end
  end

  defp age_single_tool_result(result, _distance, nil, _input), do: {[result], []}

  defp age_single_tool_result(result, distance, tool_def, input) do
    cond do
      pruned?(distance, tool_def) ->
        {[], []}

      expired?(distance, tool_def) ->
        {[render_expired(result, tool_def, input)], []}

      true ->
        render_active(result, distance, tool_def)
    end
  end

  defp pruned?(distance, tool_def) do
    case Map.get(tool_def, :pruning_distance) do
      nil -> false
      :infinity -> false
      pd -> distance >= pd
    end
  end

  defp expired?(distance, tool_def) do
    case Map.get(tool_def, :expiry_distance) do
      nil -> false
      ed -> distance >= ed
    end
  end

  defp render_active(result, distance, tool_def) do
    case {distance, Map.get(tool_def, :tail_hints)} do
      {_, nil} -> {[result], []}
      {0, hints} -> {[result], [%{type: "text", text: hints}]}
      _ -> {[result], []}
    end
  end

  defp render_expired(result, tool_def, input) do
    case Map.get(tool_def, :expiry_message) do
      nil ->
        result

      template ->
        rendered = EEx.eval_string(template, assigns: [result: result.content, input: input])
        %{result | content: rendered}
    end
  end
end
