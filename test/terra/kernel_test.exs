defmodule Terra.KernelTest do
  use ExUnit.Case, async: true

  alias Terra.Kernel
  alias Terra.Document

  # ── Start / Init ────────────────────────────────────────

  describe "start_link/1" do
    test "starts with declared buffer slots" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan, :analysis])
      assert Process.alive?(pid)
    end

    test "all slots initialize to nil" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan, :analysis])

      assert Kernel.read(pid, :plan) == nil
      assert Kernel.read(pid, :analysis) == nil
    end

    test "reading undeclared slot returns error" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])

      assert Kernel.read(pid, :unknown) == {:error, :unknown_slot}
    end
  end

  # ── Read / Write ────────────────────────────────────────

  describe "read/write" do
    test "write stores a document, read returns it with buffer wrapper" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])
      doc = Document.new("facts", "Extracted facts", "User prefers Celsius")

      assert :ok = Kernel.write(pid, :plan, doc)
      result = Kernel.read(pid, :plan)
      assert result.title == doc.title
      assert result.context == doc.context
      assert result.content =~ "<plan-buffer state=\"unlocked\""
      assert result.content =~ "updated-at="
      assert result.content =~ "User prefers Celsius"
      assert result.content =~ "</plan-buffer>"
    end

    test "write overwrites previous document" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])
      doc1 = Document.new("facts", "v1", "old")
      doc2 = Document.new("facts", "v2", "new")

      Kernel.write(pid, :plan, doc1)
      Kernel.write(pid, :plan, doc2)

      result = Kernel.read(pid, :plan)
      assert result.content =~ "new"
      refute result.content =~ "\nold\n"
    end

    test "write to undeclared slot returns error" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])

      assert Kernel.write(pid, :unknown, Document.new("x", "x", "x")) ==
               {:error, :unknown_slot}
    end

    test "write nil clears the slot" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])
      Kernel.write(pid, :plan, Document.new("x", "x", "x"))
      Kernel.write(pid, :plan, nil)

      assert Kernel.read(pid, :plan) == nil
    end
  end

  # ── Snapshot ────────────────────────────────────────────

  describe "snapshot/1" do
    test "returns all slots as a map" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan, :analysis])
      doc = Document.new("facts", "Facts", "data")
      Kernel.write(pid, :plan, doc)

      snapshot = Kernel.snapshot(pid)

      assert snapshot == %{plan: doc, analysis: nil}
    end
  end

  # ── Locking ─────────────────────────────────────────────

  describe "lock/unlock" do
    test "lock succeeds on unlocked slot" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])

      assert :ok = Kernel.lock(pid, :plan)
    end

    test "lock fails if already locked" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])
      :ok = Kernel.lock(pid, :plan)

      assert {:error, :already_locked} = Kernel.lock(pid, :plan)
    end

    test "unlock releases the lock" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])
      :ok = Kernel.lock(pid, :plan)
      :ok = Kernel.unlock(pid, :plan)

      # Can lock again
      assert :ok = Kernel.lock(pid, :plan)
    end

    test "write while locked returns error" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])
      :ok = Kernel.lock(pid, :plan)

      assert {:error, :locked} =
               Kernel.write(pid, :plan, Document.new("x", "x", "x"))
    end

    test "write_and_unlock atomically writes and releases lock" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])
      doc = Document.new("facts", "Facts", "new data")

      :ok = Kernel.lock(pid, :plan)
      :ok = Kernel.write_and_unlock(pid, :plan, doc)

      result = Kernel.read(pid, :plan)
      assert result.content =~ "new data"
      assert result.content =~ "state=\"unlocked\""
      # Lock released — can lock again
      assert :ok = Kernel.lock(pid, :plan)
    end

    test "read shows locked state when slot is locked" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])
      doc = Document.new("facts", "Facts", "data")
      Kernel.write(pid, :plan, doc)
      Kernel.lock(pid, :plan)

      result = Kernel.read(pid, :plan)
      assert result.content =~ "state=\"locked\""
    end

    test "lock on undeclared slot returns error" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])

      assert {:error, :unknown_slot} = Kernel.lock(pid, :unknown)
    end

    test "lock auto-released when locking process dies" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])

      # Lock from a separate process that dies
      task =
        Task.async(fn ->
          :ok = Kernel.lock(pid, :plan)
          :ok
        end)

      Task.await(task)
      # Give kernel time to process DOWN
      Process.sleep(10)

      # Lock should be released
      assert :ok = Kernel.lock(pid, :plan)
    end
  end

  # ── Status / Metadata ───────────────────────────────────

  describe "status/2" do
    test "reports idle slot with empty metadata" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])

      assert {:ok, status} = Kernel.status(pid, :plan)
      assert status.state == :idle
      assert status.updated_at == nil
      assert status.size == 0
      assert status.metadata == %{}
    end

    test "reports locked state and content size after write" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])
      doc = Document.new("facts", "ctx", "hello")

      :ok = Kernel.lock(pid, :plan)
      :ok = Kernel.write_and_unlock(pid, :plan, doc)
      :ok = Kernel.lock(pid, :plan)

      assert {:ok, status} = Kernel.status(pid, :plan)
      assert status.state == :locked
      assert status.updated_at != nil
      assert status.size == String.length("hello")
    end

    test "status on undeclared slot returns error" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])

      assert {:error, :unknown_slot} = Kernel.status(pid, :unknown)
    end
  end

  describe "lock/3 and set_metadata/3" do
    test "lock merges caller-supplied metadata, readable via status" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])

      :ok = Kernel.lock(pid, :plan, %{focus: "intro", request_id: "abc"})

      assert {:ok, %{metadata: %{focus: "intro", request_id: "abc"}}} =
               Kernel.status(pid, :plan)
    end

    test "lock with empty metadata leaves existing metadata untouched" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])

      :ok = Kernel.set_metadata(pid, :plan, %{focus: "intro"})
      :ok = Kernel.lock(pid, :plan, %{})

      assert {:ok, %{metadata: %{focus: "intro"}}} = Kernel.status(pid, :plan)
    end

    test "set_metadata shallow-merges without touching the lock" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])

      :ok = Kernel.set_metadata(pid, :plan, %{focus: "intro", step: 1})
      :ok = Kernel.set_metadata(pid, :plan, %{step: 2})

      assert {:ok, status} = Kernel.status(pid, :plan)
      assert status.state == :idle
      assert status.metadata == %{focus: "intro", step: 2}
    end

    test "metadata persists across lock release" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])
      doc = Document.new("d", "c", "body")

      :ok = Kernel.lock(pid, :plan, %{focus: "intro"})
      :ok = Kernel.write_and_unlock(pid, :plan, doc)

      assert {:ok, %{state: :idle, metadata: %{focus: "intro"}}} =
               Kernel.status(pid, :plan)
    end

    test "set_metadata on undeclared slot returns error" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])

      assert {:error, :unknown_slot} = Kernel.set_metadata(pid, :unknown, %{a: 1})
    end
  end

  # ── Subscriptions / Notifications ───────────────────────

  describe "subscribe/notify" do
    test "subscriber receives notification on write" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])
      Kernel.subscribe(pid)

      doc = Document.new("facts", "Facts", "data")
      Kernel.write(pid, :plan, doc)

      assert_receive {:kernel_update, :plan, ^doc}
    end

    test "subscriber receives notification on write_and_unlock" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])
      Kernel.subscribe(pid)

      doc = Document.new("facts", "Facts", "data")
      Kernel.lock(pid, :plan)
      Kernel.write_and_unlock(pid, :plan, doc)

      assert_receive {:kernel_update, :plan, ^doc}
    end

    test "multiple subscribers all receive notifications" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])

      # Subscribe from two processes
      parent = self()

      pids =
        for i <- 1..2 do
          spawn(fn ->
            Kernel.subscribe(pid)
            send(parent, {:subscribed, i})

            receive do
              msg -> send(parent, {:got, i, msg})
            end
          end)
        end

      # Wait for subscriptions
      assert_receive {:subscribed, 1}
      assert_receive {:subscribed, 2}

      doc = Document.new("facts", "Facts", "data")
      Kernel.write(pid, :plan, doc)

      assert_receive {:got, 1, {:kernel_update, :plan, ^doc}}
      assert_receive {:got, 2, {:kernel_update, :plan, ^doc}}

      # Cleanup
      for p <- pids, do: Process.exit(p, :kill)
    end

    test "dead subscriber is automatically removed" do
      {:ok, pid} = Kernel.start_link(buffers: [:plan])

      # Subscribe from a process that dies
      task =
        Task.async(fn ->
          Kernel.subscribe(pid)
          :ok
        end)

      Task.await(task)
      Process.sleep(10)

      # Write should not crash (dead subscriber cleaned up)
      doc = Document.new("facts", "Facts", "data")
      assert :ok = Kernel.write(pid, :plan, doc)
    end
  end
end
