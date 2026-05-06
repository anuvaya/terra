defmodule Terra.APIError do
  @moduledoc """
  Structured error type for LLM API errors.

  Provides typed errors with `request_id` and `retry_after` for debugging and
  retry logic. All three providers (`Anthropic`, `OpenAI`, `Google`) emit
  `Terra.APIError` structs on failure.

  ## Error Types

  | Type                | Status | Retryable? |
  |---------------------|--------|------------|
  | `:invalid_request`  | 400    | No         |
  | `:authentication`   | 401    | No         |
  | `:permission`       | 403    | No         |
  | `:not_found`        | 404    | No         |
  | `:rate_limit`       | 429    | Yes        |
  | `:api_error`        | 5xx    | Yes        |
  | `:overloaded`       | 529    | Yes        |

  ## Retry Logic

      case error do
        %Terra.APIError{} = e ->
          if Terra.APIError.retryable?(e) do
            delay = e.retry_after || 1000
            Process.sleep(delay)
            # retry...
          end
      end
  """
  defexception [:status, :type, :message, :request_id, :retry_after]

  @type error_type ::
          :invalid_request
          | :authentication
          | :permission
          | :not_found
          | :rate_limit
          | :api_error
          | :overloaded
          | :stream_error
          | :unknown

  @impl true
  def message(%__MODULE__{} = error) do
    status_str = if error.status, do: " (#{error.status})", else: ""
    req_id_str = if error.request_id, do: " [#{error.request_id}]", else: ""
    "#{error.type}#{status_str}: #{error.message}#{req_id_str}"
  end

  @doc """
  Create error from a Req response struct.
  """
  def from_response(%{status: status, headers: headers} = resp) do
    body = Map.get(resp, :body, "")
    request_id = get_header(headers, "x-request-id")
    retry_after = parse_retry_after(headers)
    {type, message} = extract_error(status, body)

    struct(__MODULE__,
      status: status,
      type: type,
      message: message,
      request_id: request_id,
      retry_after: retry_after
    )
  end

  def from_response(%{status: status, body: body}) do
    {type, message} = extract_error(status, body)
    struct(__MODULE__, status: status, type: type, message: message)
  end

  @doc """
  Create error from raw body string (e.g. accumulated during streaming).
  """
  def from_raw_body(status, raw, headers \\ []) do
    request_id = get_header(headers, "x-request-id")
    retry_after = parse_retry_after(headers)
    {type, message} = parse_error_body(status, raw)

    struct(__MODULE__,
      status: status,
      type: type,
      message: message,
      request_id: request_id,
      retry_after: retry_after
    )
  end

  @doc """
  Create error from an SSE error event during streaming.
  """
  def from_sse_event(%{"error" => %{"type" => type, "message" => message}}) do
    struct(__MODULE__, type: type_from_string(type), message: message)
  end

  def from_sse_event(%{"type" => type, "message" => message}) do
    struct(__MODULE__, type: type_from_string(type), message: message)
  end

  @doc """
  Check if this error type is retryable.
  """
  def retryable?(%__MODULE__{type: type}) do
    type in [:rate_limit, :api_error, :overloaded]
  end

  # ── Private ────────────────────────────────────────────

  defp extract_error(status, body) when is_binary(body) and body != "" do
    parse_error_body(status, body)
  end

  defp extract_error(status, %{"error" => %{"type" => type, "message" => message}}) do
    {resolve_type(status, type), message}
  end

  defp extract_error(status, _body) do
    {type_from_status(status), default_message(status)}
  end

  defp parse_error_body(status, raw) do
    case Jason.decode(raw) do
      {:ok, %{"error" => %{"type" => type, "message" => message}}} ->
        {resolve_type(status, type), message}

      {:ok, %{"type" => "error", "error" => %{"type" => type, "message" => message}}} ->
        {resolve_type(status, type), message}

      {:ok, %{"error" => %{"message" => message}}} ->
        {type_from_status(status), message}

      _ ->
        {type_from_status(status), if(raw == "", do: default_message(status), else: raw)}
    end
  end

  # HTTP status is authoritative for unambiguous categories. Some providers
  # (notably OpenAI) return 401 with body type "invalid_request_error" for
  # malformed keys — the status code is the source of truth.
  defp resolve_type(status, _type) when status in [401, 403, 404, 429, 529] do
    type_from_status(status)
  end

  defp resolve_type(_status, type), do: type_from_string(type)

  defp type_from_string("invalid_request_error"), do: :invalid_request
  defp type_from_string("authentication_error"), do: :authentication
  defp type_from_string("permission_error"), do: :permission
  defp type_from_string("not_found_error"), do: :not_found
  defp type_from_string("rate_limit_error"), do: :rate_limit
  defp type_from_string("api_error"), do: :api_error
  defp type_from_string("overloaded_error"), do: :overloaded
  defp type_from_string(_), do: :unknown

  defp type_from_status(400), do: :invalid_request
  defp type_from_status(401), do: :authentication
  defp type_from_status(403), do: :permission
  defp type_from_status(404), do: :not_found
  defp type_from_status(429), do: :rate_limit
  defp type_from_status(status) when status in [500, 502, 503], do: :api_error
  defp type_from_status(529), do: :overloaded
  defp type_from_status(_), do: :unknown

  defp default_message(400), do: "Bad request"
  defp default_message(401), do: "Invalid API key"
  defp default_message(403), do: "Access denied"
  defp default_message(404), do: "Resource not found"
  defp default_message(429), do: "Rate limited"
  defp default_message(500), do: "Internal server error"
  defp default_message(502), do: "Bad gateway"
  defp default_message(503), do: "Service unavailable"
  defp default_message(529), do: "API is overloaded"
  defp default_message(_), do: "Unknown error"

  defp get_header(headers, name) when is_list(headers) do
    name_lower = String.downcase(name)

    Enum.find_value(headers, fn
      {key, value} when is_binary(key) ->
        if String.downcase(key) == name_lower, do: value
      _ -> nil
    end)
  end

  defp get_header(_, _), do: nil

  defp parse_retry_after(headers) when is_list(headers) do
    case get_header(headers, "retry-after") do
      nil -> nil
      value ->
        case Integer.parse(value) do
          {seconds, _} -> seconds * 1000
          :error -> nil
        end
    end
  end

  defp parse_retry_after(_), do: nil
end
