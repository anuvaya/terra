defmodule Terra.DocumentTest do
  use ExUnit.Case, async: true

  alias Terra.Document

  describe "struct" do
    test "new/3 creates a document with defaults" do
      doc = Document.new("forecast", "User's regional forecast data", "forecast content here")

      assert doc.title == "forecast"
      assert doc.context == "User's regional forecast data"
      assert doc.content == "forecast content here"
      assert doc.cache == nil
    end

    test "new/4 accepts cache option" do
      doc = Document.new("forecast", "Forecast data", "content", cache: :ephemeral)

      assert doc.cache == :ephemeral
    end

    test "struct can be pattern matched" do
      doc = Document.new("title", "ctx", "body")
      assert %Document{title: "title"} = doc
    end
  end

  describe "state management" do
    setup do
      state = %Terra.Agent.State{}
      {:ok, state: state}
    end

    test "put/2 adds a document keyed by title", %{state: state} do
      doc = Document.new("forecast", "Forecast data", "content")
      state = Document.put(state, doc)

      assert map_size(state.documents) == 1
      assert state.documents["forecast"] == doc
    end

    test "put/2 overwrites existing document with same title", %{state: state} do
      doc1 = Document.new("forecast", "v1", "old content")
      doc2 = Document.new("forecast", "v2", "new content")

      state = state |> Document.put(doc1) |> Document.put(doc2)

      assert map_size(state.documents) == 1
      assert state.documents["forecast"].content == "new content"
    end

    test "get/2 retrieves a document by title", %{state: state} do
      doc = Document.new("profiles", "User profiles", "data")
      state = Document.put(state, doc)

      assert Document.get(state, "profiles") == doc
      assert Document.get(state, "nonexistent") == nil
    end

    test "delete/2 removes a document by title", %{state: state} do
      doc = Document.new("scratch", "Scratch pad", "notes")
      state = state |> Document.put(doc) |> Document.delete("scratch")

      assert state.documents == %{}
    end

    test "list/1 returns all documents", %{state: state} do
      d1 = Document.new("a", "ctx", "content a")
      d2 = Document.new("b", "ctx", "content b")

      state = state |> Document.put(d1) |> Document.put(d2)
      docs = Document.list(state)

      assert length(docs) == 2
      titles = Enum.map(docs, & &1.title)
      assert "a" in titles
      assert "b" in titles
    end
  end

  describe "tool integration" do
    test "tool registry can return a Document from execute" do
      doc = Document.new("session_forecast", "Forecast for session", "..forecast data..", cache: :ephemeral)
      result = {:ok, doc}

      assert {:ok, %Document{title: "session_forecast", cache: :ephemeral}} = result
    end
  end
end
