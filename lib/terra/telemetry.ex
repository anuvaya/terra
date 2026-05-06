defmodule Terra.Telemetry do
  @moduledoc """
  Telemetry events for Terra agents and kernels.

  Terra emits `:telemetry` events at key lifecycle points. These are
  zero-cost when no handler is attached (~50ns ETS lookup). Consumers
  attach handlers to observe agent behaviour without modifying Terra.

  ## Agent Events

  * `[:terra, :agent, :init]` — agent initialized
    * Metadata: `handler`, `state_name`, `pid`, `data`

  * `[:terra, :agent, :input]` — input received via `send_input/2`
    * Metadata: `handler`, `state_name`, `input`, `pid`, `req_id`

  * `[:terra, :agent, :state_change]` — state machine transitioned
    * Metadata: `handler`, `from`, `to`, `pid`, `req_id`

  * `[:terra, :agent, :invoke, :start]` — provider stream starting
    * Measurements: `system_time`
    * Metadata: `handler`, `state_name`, `pid`, `req_id`, `context` (message_count, tool_count, tool_names, turn_count, has_system, max_tokens)

  * `[:terra, :agent, :invoke, :stop]` — provider stream completed
    * Measurements: `duration` (native units)
    * Metadata: `handler`, `state_name`, `stop_reason`, `usage`, `pid`, `req_id`, `response` (tool_names, block_types, tool_count, turn_count)

  * `[:terra, :agent, :cancel_stream]` — active stream cancelled (user sent new input)
    * Measurements: `system_time`
    * Metadata: `handler`, `state_name`, `pid`, `req_id`

  * `[:terra, :agent, :invoke, :error]` — provider or stream error
    * Measurements: `system_time`
    * Metadata: `handler`, `error_type`, `reason`, `pid`, `req_id`

  * `[:terra, :agent, :tool, :start]` — tool execution starting
    * Measurements: `system_time`
    * Metadata: `tool_name`, `registry`, `pid`, `req_id`

  * `[:terra, :agent, :tool, :stop]` — tool execution completed
    * Measurements: `duration` (native units)
    * Metadata: `tool_name`, `registry`, `result_type`, `pid`, `req_id`

  ## Provider Events

  * `[:terra, :provider, :request]` — HTTP request about to be sent
    * Metadata: `caller_pid`, `body`, `headers`

  ## Kernel Events

  * `[:terra, :kernel, :write]` — buffer written
    * Metadata: `slot`, `doc`, `size`, `kernel_pid`, `caller_pid`

  * `[:terra, :kernel, :lock]` — buffer locked
    * Metadata: `slot`, `kernel_pid`, `caller_pid`

  * `[:terra, :kernel, :unlock]` — buffer unlocked
    * Metadata: `slot`, `kernel_pid`

  * `[:terra, :kernel, :notify]` — subscribers notified
    * Metadata: `slot`, `kernel_pid`, `subscriber_count`
  """

  # -- Agent Events --

  @doc false
  def agent_init(handler, state_name, data) do
    :telemetry.execute(
      [:terra, :agent, :init],
      %{system_time: System.system_time()},
      %{handler: handler, state_name: state_name, pid: self(), data: data}
    )
  end

  @doc false
  def agent_input(handler, state_name, input, req_id) do
    :telemetry.execute(
      [:terra, :agent, :input],
      %{system_time: System.system_time()},
      %{handler: handler, state_name: state_name, input: input, pid: self(), req_id: req_id}
    )
  end

  @doc false
  def maybe_state_change(current_state_name, translated, state) do
    req_id = state._req_id
    case translated do
      {:next_state, new_state_name, _, _} when new_state_name != current_state_name ->
        :telemetry.execute(
          [:terra, :agent, :state_change],
          %{system_time: System.system_time()},
          %{handler: state.handler, from: current_state_name, to: new_state_name, pid: self(), req_id: req_id}
        )

      {:next_state, new_state_name, _} when new_state_name != current_state_name ->
        :telemetry.execute(
          [:terra, :agent, :state_change],
          %{system_time: System.system_time()},
          %{handler: state.handler, from: current_state_name, to: new_state_name, pid: self(), req_id: req_id}
        )

      _ ->
        :ok
    end
  end

  @doc false
  def invoke_start(handler, state_name, context_meta, req_id) do
    :telemetry.execute(
      [:terra, :agent, :invoke, :start],
      %{system_time: System.system_time()},
      %{handler: handler, state_name: state_name, pid: self(), context: context_meta, req_id: req_id}
    )
  end

  @doc false
  def invoke_stop(handler, state_name, stop_reason, usage, duration, response_meta, req_id) do
    :telemetry.execute(
      [:terra, :agent, :invoke, :stop],
      %{duration: duration},
      %{handler: handler, state_name: state_name, stop_reason: stop_reason, usage: usage, pid: self(), response: response_meta, req_id: req_id}
    )
  end

  @doc false
  def cancel_stream(handler, state_name, req_id) do
    :telemetry.execute(
      [:terra, :agent, :cancel_stream],
      %{system_time: System.system_time()},
      %{handler: handler, state_name: state_name, pid: self(), req_id: req_id}
    )
  end

  @doc false
  def invoke_error(handler, error_type, reason, req_id) do
    :telemetry.execute(
      [:terra, :agent, :invoke, :error],
      %{system_time: System.system_time()},
      %{handler: handler, error_type: error_type, reason: reason, pid: self(), req_id: req_id}
    )
  end

  @doc false
  def tool_start(tool_name, registry, req_id) do
    :telemetry.execute(
      [:terra, :agent, :tool, :start],
      %{system_time: System.system_time()},
      %{tool_name: tool_name, registry: registry, pid: self(), req_id: req_id}
    )
  end

  @doc false
  def tool_stop(tool_name, registry, result_type, duration, req_id) do
    :telemetry.execute(
      [:terra, :agent, :tool, :stop],
      %{duration: duration},
      %{tool_name: tool_name, registry: registry, result_type: result_type, pid: self(), req_id: req_id}
    )
  end

  # -- Provider Events --

  @doc false
  def provider_request(caller_pid, body, headers) do
    :telemetry.execute(
      [:terra, :provider, :request],
      %{system_time: System.system_time()},
      %{caller_pid: caller_pid, body: body, headers: headers}
    )
  end

  # -- Kernel Events --

  @doc false
  def kernel_write(slot, doc, caller_pid) do
    size = if doc, do: byte_size(doc.content || ""), else: 0

    :telemetry.execute(
      [:terra, :kernel, :write],
      %{system_time: System.system_time()},
      %{slot: slot, doc: doc, size: size, kernel_pid: self(), caller_pid: caller_pid}
    )
  end

  @doc false
  def kernel_lock(slot, caller_pid) do
    :telemetry.execute(
      [:terra, :kernel, :lock],
      %{system_time: System.system_time()},
      %{slot: slot, kernel_pid: self(), caller_pid: caller_pid}
    )
  end

  @doc false
  def kernel_unlock(slot) do
    :telemetry.execute(
      [:terra, :kernel, :unlock],
      %{system_time: System.system_time()},
      %{slot: slot, kernel_pid: self()}
    )
  end

  @doc false
  def kernel_notify(slot, subscriber_count) do
    :telemetry.execute(
      [:terra, :kernel, :notify],
      %{system_time: System.system_time()},
      %{slot: slot, kernel_pid: self(), subscriber_count: subscriber_count}
    )
  end
end
