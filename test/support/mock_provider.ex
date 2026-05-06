defmodule Terra.Test.MockProvider do
  @moduledoc """
  Deterministic mock provider for testing. Replays Anthropic-shaped
  stream events to the caller process.
  """

  @behaviour Terra.Provider

  @impl true
  def stream(caller, _params) do
    ref = make_ref()
    events = Process.get(:mock_provider_events, default_events())

    spawn_link(fn ->
      for event <- events do
        send(caller, {:stream, ref, event})
      end
    end)

    {:ok, ref}
  end

  @impl true
  def cancel(_ref) do
    :ok
  end

  def default_events do
    [
      {:message_start,
       %{
         id: "msg_test",
         type: "message",
         role: "assistant",
         content: [],
         usage: %{input_tokens: 10, output_tokens: 0}
       }},
      {:content_block_start, 0, %{type: "text", text: ""}},
      {:content_block_delta, 0, %{type: "text_delta", text: "Hello"}},
      {:content_block_delta, 0, %{type: "text_delta", text: " world"}},
      {:content_block_stop, 0},
      {:message_delta, %{stop_reason: "end_turn"}, %{output_tokens: 5}},
      :message_stop
    ]
  end
end
