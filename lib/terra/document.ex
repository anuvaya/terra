defmodule Terra.Document do
  @moduledoc """
  A structured content block for injecting into LLM context.

  Documents represent large, stable content (forecasts, profiles, cached data)
  that benefit from prompt caching and structured rendering. They can appear in:

  - **Tool results** — when a `Terra.Tool` has `result_type: :document`, the
    `Terra.ToolRegistry` `execute/3` returns `{:ok, %Terra.Document{}}` and
    the provider serializes it as a document block (enabling prompt caching)
  - **Context building** — consumer injects documents via `Terra.Context.document/2`
    into the first user message
  - **Agent state** — documents can be stored on the `Terra.Agent.State` via
    `put/2` and retrieved with `get/2` for reuse across turns

  ## Fields

  - `title` — identifier for the document block
  - `context` — description shown to the LLM (what this document contains)
  - `content` — the actual text payload
  - `cache` — `:ephemeral` or `nil` (controls provider-level prompt caching)
  """

  @type t :: %__MODULE__{
          title: String.t(),
          context: String.t(),
          content: String.t(),
          cache: :ephemeral | nil
        }

  defstruct [:title, :context, :content, cache: nil]

  @doc """
  Create a new document.

  ## Options

    * `:cache` — set to `:ephemeral` to enable provider-level prompt caching
  """
  @spec new(String.t(), String.t(), String.t(), keyword()) :: t()
  def new(title, context, content, opts \\ []) do
    %__MODULE__{
      title: title,
      context: context,
      content: content,
      cache: Keyword.get(opts, :cache)
    }
  end

  @doc """
  Store a document on the agent state, keyed by its `title`.

  Stored documents can be injected into context via `Terra.Context.document/2`
  or read back with `get/2`.
  """
  @spec put(Terra.Agent.State.t(), t()) :: Terra.Agent.State.t()
  def put(%Terra.Agent.State{} = state, %__MODULE__{} = doc) do
    %{state | documents: Map.put(state.documents, doc.title, doc)}
  end

  @doc """
  Retrieve a document from the agent state by `title`. Returns `nil` if not found.
  """
  @spec get(Terra.Agent.State.t(), String.t()) :: t() | nil
  def get(%Terra.Agent.State{} = state, title) do
    Map.get(state.documents, title)
  end

  @doc """
  Remove a document from the agent state by `title`.
  """
  @spec delete(Terra.Agent.State.t(), String.t()) :: Terra.Agent.State.t()
  def delete(%Terra.Agent.State{} = state, title) do
    %{state | documents: Map.delete(state.documents, title)}
  end

  @doc """
  List all documents stored on the agent state.
  """
  @spec list(Terra.Agent.State.t()) :: [t()]
  def list(%Terra.Agent.State{} = state) do
    Map.values(state.documents)
  end
end
