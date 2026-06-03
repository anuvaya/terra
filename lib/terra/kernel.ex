defmodule Terra.Kernel do
  @moduledoc """
  Shared document store for multi-agent sessions.

  The kernel holds named buffer slots that multiple agents can read from
  and write to. It provides locking to serialize writes and notifies
  subscribers when slots are updated.

  ## Usage

      {:ok, kernel} = Terra.Kernel.start_link(buffers: [:plan, :analysis])

      # Read (non-blocking, always succeeds)
      doc = Terra.Kernel.read(kernel, :plan)

      # Write (notifies subscribers)
      Terra.Kernel.write(kernel, :plan, %Terra.Document{...})

      # Lock → work → write_and_unlock (atomic)
      :ok = Terra.Kernel.lock(kernel, :plan)
      :ok = Terra.Kernel.write_and_unlock(kernel, :plan, doc)

      # Subscribe to changes
      Terra.Kernel.subscribe(kernel)
      # receive {:kernel_update, :plan, doc} -> ...
  """

  use GenServer

  # ── Public API ──────────────────────────────────────────

  def start_link(opts) do
    buffers = Keyword.fetch!(opts, :buffers)
    GenServer.start_link(__MODULE__, buffers, Keyword.drop(opts, [:buffers]))
  end

  @doc """
  Read the document in `slot`. Returns `nil` if the slot is empty,
  or `{:error, :unknown_slot}` if the slot doesn't exist.
  """
  @spec read(GenServer.server(), atom()) :: Terra.Document.t() | nil | {:error, :unknown_slot}
  def read(kernel, slot) do
    GenServer.call(kernel, {:read, slot})
  end

  @doc """
  Read the raw document from `slot` without kernel wrapper tags.
  """
  @spec read_raw(GenServer.server(), atom()) :: Terra.Document.t() | nil | {:error, :unknown_slot}
  def read_raw(kernel, slot) do
    GenServer.call(kernel, {:read_raw, slot})
  end

  @doc """
  Write a document to `slot`, notifying all subscribers.

  Returns `{:error, :locked}` if another process holds a lock on the slot.
  """
  @spec write(GenServer.server(), atom(), Terra.Document.t() | nil) ::
          :ok | {:error, :unknown_slot | :locked}
  def write(kernel, slot, doc) do
    GenServer.call(kernel, {:write, slot, doc})
  end

  @doc """
  Atomically write a document and release the lock on `slot`.

  Use this after `lock/2` to ensure the write and unlock happen as one
  operation, preventing race conditions with other writers.
  """
  @spec write_and_unlock(GenServer.server(), atom(), Terra.Document.t() | nil) ::
          :ok | {:error, :unknown_slot | :not_locked}
  def write_and_unlock(kernel, slot, doc) do
    GenServer.call(kernel, {:write_and_unlock, slot, doc})
  end

  @doc """
  Return a snapshot of all slots as `%{slot_name => document | nil}`.
  """
  @spec snapshot(GenServer.server()) :: %{atom() => Terra.Document.t() | nil}
  def snapshot(kernel) do
    GenServer.call(kernel, :snapshot)
  end

  @doc """
  Acquire an exclusive lock on `slot`, optionally merging caller-supplied
  metadata into the slot's metadata map.

  Returns `{:error, :already_locked}` if another process already holds it.
  Locks are automatically released if the locking process exits. The metadata
  map is opaque to the kernel — callers attach whatever context they want
  (working focus, intent, request id, etc.) and read it back via `status/2`.
  Passed metadata is shallow-merged into existing metadata; pass `%{}` to
  leave it untouched.
  """
  @spec lock(GenServer.server(), atom(), map()) ::
          :ok | {:error, :unknown_slot | :already_locked}
  def lock(kernel, slot, metadata \\ %{}) when is_map(metadata) do
    GenServer.call(kernel, {:lock, slot, metadata})
  end

  @doc """
  Merge `metadata` into the slot's metadata map without touching the lock.

  Used by callers that refine slot context mid-loop. Shallow merge: keys
  present in `metadata` overwrite existing ones; everything else is preserved.
  """
  @spec set_metadata(GenServer.server(), atom(), map()) :: :ok | {:error, :unknown_slot}
  def set_metadata(kernel, slot, metadata) when is_map(metadata) do
    GenServer.call(kernel, {:set_metadata, slot, metadata})
  end

  @doc """
  Release the lock on `slot`. No-op if the slot is already unlocked.
  """
  @spec unlock(GenServer.server(), atom()) :: :ok | {:error, :unknown_slot}
  def unlock(kernel, slot) do
    GenServer.call(kernel, {:unlock, slot})
  end

  @doc """
  Return lightweight status for `slot`: locked/idle, updated_at, content size,
  and the caller-supplied metadata map.
  """
  @spec status(GenServer.server(), atom()) :: {:ok, map()} | {:error, :unknown_slot}
  def status(kernel, slot) do
    GenServer.call(kernel, {:status, slot})
  end

  @doc """
  Subscribe the calling process to slot updates.

  The subscriber receives `{:kernel_update, slot, document}` messages
  whenever a slot is written to. The subscription is automatically
  cleaned up when the subscriber exits.
  """
  @spec subscribe(GenServer.server()) :: :ok
  def subscribe(kernel) do
    GenServer.call(kernel, {:subscribe, self()})
  end

  # ── GenServer ───────────────────────────────────────────

  @impl true
  def init(buffers) do
    slots =
      Map.new(buffers, fn name ->
        {name, %{doc: nil, locked_by: nil, updated_at: nil, metadata: %{}}}
      end)

    {:ok, %{slots: slots, subscribers: %{}}}
  end

  @impl true
  def handle_call({:read, slot}, _from, state) do
    case Map.get(state.slots, slot) do
      nil ->
        {:reply, {:error, :unknown_slot}, state}

      %{doc: nil} ->
        {:reply, nil, state}

      %{doc: %Terra.Document{} = doc, locked_by: locked_by, updated_at: updated_at} ->
        lock_state = if locked_by != nil, do: "locked", else: "unlocked"

        updated_attr =
          if updated_at, do: " updated-at=\"#{DateTime.to_iso8601(updated_at)}\"", else: ""

        wrapped_content =
          "<#{slot}-buffer state=\"#{lock_state}\"#{updated_attr}>\n#{doc.content}\n</#{slot}-buffer>"

        {:reply, %{doc | content: wrapped_content}, state}
    end
  end

  def handle_call({:status, slot}, _from, state) do
    case Map.get(state.slots, slot) do
      nil ->
        {:reply, {:error, :unknown_slot}, state}

      %{doc: doc, locked_by: locked_by, updated_at: updated_at, metadata: metadata} ->
        {:reply,
         {:ok,
          %{
            state: if(locked_by != nil, do: :locked, else: :idle),
            updated_at: updated_at,
            size: if(doc, do: String.length(doc.content), else: 0),
            metadata: metadata
          }}, state}
    end
  end

  def handle_call({:read_raw, slot}, _from, state) do
    case Map.get(state.slots, slot) do
      nil -> {:reply, {:error, :unknown_slot}, state}
      %{doc: nil} -> {:reply, nil, state}
      %{doc: %Terra.Document{} = doc} -> {:reply, doc, state}
    end
  end

  def handle_call({:write, slot, doc}, {caller_pid, _} = _from, state) do
    case Map.get(state.slots, slot) do
      nil ->
        {:reply, {:error, :unknown_slot}, state}

      %{locked_by: pid} when pid != nil ->
        {:reply, {:error, :locked}, state}

      %{} ->
        now = DateTime.utc_now()
        slots = state.slots |> put_in([slot, :doc], doc) |> put_in([slot, :updated_at], now)
        state = %{state | slots: slots}
        Terra.Telemetry.kernel_write(slot, doc, caller_pid)
        notify_subscribers(state, slot, doc)
        {:reply, :ok, state}
    end
  end

  def handle_call({:write_and_unlock, slot, doc}, {caller_pid, _} = _from, state) do
    case Map.get(state.slots, slot) do
      nil ->
        {:reply, {:error, :unknown_slot}, state}

      %{} ->
        now = DateTime.utc_now()

        slots =
          state.slots
          |> put_in([slot, :doc], doc)
          |> put_in([slot, :locked_by], nil)
          |> put_in([slot, :updated_at], now)

        state = %{state | slots: slots}
        Terra.Telemetry.kernel_write(slot, doc, caller_pid)
        Terra.Telemetry.kernel_unlock(slot)
        notify_subscribers(state, slot, doc)
        {:reply, :ok, state}
    end
  end

  def handle_call(:snapshot, _from, state) do
    snapshot = Map.new(state.slots, fn {name, %{doc: doc}} -> {name, doc} end)
    {:reply, snapshot, state}
  end

  def handle_call({:lock, slot, metadata}, {from_pid, _}, state) do
    case Map.get(state.slots, slot) do
      nil ->
        {:reply, {:error, :unknown_slot}, state}

      %{locked_by: pid} when pid != nil ->
        {:reply, {:error, :already_locked}, state}

      %{metadata: current_metadata} ->
        ref = Process.monitor(from_pid)

        slots =
          state.slots
          |> put_in([slot, :locked_by], {from_pid, ref})
          |> put_in([slot, :metadata], Map.merge(current_metadata, metadata))

        Terra.Telemetry.kernel_lock(slot, from_pid)
        {:reply, :ok, %{state | slots: slots}}
    end
  end

  def handle_call({:set_metadata, slot, metadata}, _from, state) do
    case Map.get(state.slots, slot) do
      nil ->
        {:reply, {:error, :unknown_slot}, state}

      %{metadata: current_metadata} ->
        slots = put_in(state.slots, [slot, :metadata], Map.merge(current_metadata, metadata))
        {:reply, :ok, %{state | slots: slots}}
    end
  end

  def handle_call({:unlock, slot}, _from, state) do
    case Map.get(state.slots, slot) do
      nil ->
        {:reply, {:error, :unknown_slot}, state}

      %{locked_by: {_pid, ref}} when ref != nil ->
        Process.demonitor(ref, [:flush])
        slots = put_in(state.slots, [slot, :locked_by], nil)
        Terra.Telemetry.kernel_unlock(slot)
        {:reply, :ok, %{state | slots: slots}}

      %{} ->
        {:reply, :ok, state}
    end
  end

  def handle_call({:subscribe, pid}, _from, state) do
    ref = Process.monitor(pid)
    subscribers = Map.put(state.subscribers, pid, ref)
    {:reply, :ok, %{state | subscribers: subscribers}}
  end

  @impl true
  def handle_info({:DOWN, _ref, :process, pid, _reason}, state) do
    # Clean up subscriber
    state =
      case Map.pop(state.subscribers, pid) do
        {nil, _} -> state
        {_ref, subscribers} -> %{state | subscribers: subscribers}
      end

    # Release any locks held by this process
    slots =
      Map.new(state.slots, fn
        {name, %{locked_by: {^pid, _ref}} = slot} ->
          {name, %{slot | locked_by: nil}}

        entry ->
          entry
      end)

    {:noreply, %{state | slots: slots}}
  end

  def handle_info(_msg, state) do
    {:noreply, state}
  end

  # ── Helpers ─────────────────────────────────────────────

  defp notify_subscribers(state, slot, doc) do
    subscriber_count = map_size(state.subscribers)

    for {pid, _ref} <- state.subscribers do
      send(pid, {:kernel_update, slot, doc})
    end

    Terra.Telemetry.kernel_notify(slot, subscriber_count)
  end
end
