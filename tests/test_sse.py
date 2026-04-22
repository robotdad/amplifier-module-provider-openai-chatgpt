"""Tests for _sse.py SSE parser."""

from __future__ import annotations

import json

import pytest

from amplifier_module_provider_openai_chatgpt._sse import (
    SSEError,
    parse_sse_events,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _line(event: dict) -> str:
    """Encode a dict as an SSE data line."""
    return f"data: {json.dumps(event)}"


def _done() -> str:
    return "data: [DONE]"


# ---------------------------------------------------------------------------
# TestParseSSEEvents
# ---------------------------------------------------------------------------


class TestParseSSEEvents:
    def test_simple_text_output_text_type(self) -> None:
        """response.output_item.done with output_text content type accumulates text."""
        lines = [
            _line(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "Hello, world!"},
                        ],
                    },
                }
            ),
            _done(),
        ]
        result = parse_sse_events(lines)
        assert result.content == "Hello, world!"
        assert result.tool_calls == []

    def test_text_content_type(self) -> None:
        """response.output_item.done with 'text' content type also accumulates text."""
        lines = [
            _line(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "content": [
                            {"type": "text", "text": "Plain text content"},
                        ],
                    },
                }
            ),
            _done(),
        ]
        result = parse_sse_events(lines)
        assert result.content == "Plain text content"
        assert result.tool_calls == []

    def test_function_call_accumulation(self) -> None:
        """response.output_item.done with function_call item extracts tool call."""
        lines = [
            _line(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_abc123",
                        "name": "get_weather",
                        "arguments": '{"city": "Seattle"}',
                    },
                }
            ),
            _done(),
        ]
        result = parse_sse_events(lines)
        assert result.content == ""
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc["id"] == "call_abc123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city": "Seattle"}'

    def test_function_call_uses_id_fallback(self) -> None:
        """When call_id is absent, use item 'id' field."""
        lines = [
            _line(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "id": "item_id_xyz",
                        "name": "do_thing",
                        "arguments": "{}",
                    },
                }
            ),
            _done(),
        ]
        result = parse_sse_events(lines)
        assert result.tool_calls[0]["id"] == "item_id_xyz"

    def test_mixed_text_and_tools(self) -> None:
        """Multiple output items accumulate text and tool calls together."""
        lines = [
            _line(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Thinking..."}],
                    },
                }
            ),
            _line(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "search",
                        "arguments": '{"q": "foo"}',
                    },
                }
            ),
            _done(),
        ]
        result = parse_sse_events(lines)
        assert result.content == "Thinking..."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "search"

    def test_non_data_lines_skipped(self) -> None:
        """Lines that don't start with 'data: ' are ignored."""
        lines = [
            "event: response.output_item.done",
            ": comment line",
            "",
            "retry: 3000",
            _line(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    },
                }
            ),
            _done(),
        ]
        result = parse_sse_events(lines)
        assert result.content == "ok"

    def test_malformed_json_skipped(self) -> None:
        """Lines with invalid JSON are skipped gracefully."""
        lines = [
            "data: {not valid json",
            "data: }also bad",
            _line(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "recovered"}],
                    },
                }
            ),
            _done(),
        ]
        result = parse_sse_events(lines)
        assert result.content == "recovered"

    def test_done_sentinel_stops_parsing(self) -> None:
        """[DONE] stops parsing; subsequent lines are ignored."""
        lines = [
            _line(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "first"}],
                    },
                }
            ),
            _done(),
            _line(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": " SHOULD NOT APPEAR"}
                        ],
                    },
                }
            ),
        ]
        result = parse_sse_events(lines)
        assert result.content == "first"

    def test_usage_from_response_done(self) -> None:
        """input_tokens and output_tokens extracted from response.done event."""
        lines = [
            _line(
                {
                    "type": "response.done",
                    "response": {
                        "id": "resp_abc",
                        "model": "gpt-4o",
                        "usage": {
                            "input_tokens": 42,
                            "output_tokens": 17,
                        },
                    },
                }
            ),
            _done(),
        ]
        result = parse_sse_events(lines)
        assert result.input_tokens == 42
        assert result.output_tokens == 17

    def test_response_id_and_model_from_response_created(self) -> None:
        """response_id and model extracted from response.created event."""
        lines = [
            _line(
                {
                    "type": "response.created",
                    "response": {
                        "id": "resp_xyz789",
                        "model": "gpt-4o-mini",
                    },
                }
            ),
            _done(),
        ]
        result = parse_sse_events(lines)
        assert result.response_id == "resp_xyz789"
        assert result.model == "gpt-4o-mini"

    def test_response_id_and_model_from_response_done(self) -> None:
        """response_id and model also available from response.done event."""
        lines = [
            _line(
                {
                    "type": "response.done",
                    "response": {
                        "id": "resp_done123",
                        "model": "gpt-4o",
                        "usage": {"input_tokens": 5, "output_tokens": 3},
                    },
                }
            ),
            _done(),
        ]
        result = parse_sse_events(lines)
        assert result.response_id == "resp_done123"
        assert result.model == "gpt-4o"

    def test_raw_events_populated(self) -> None:
        """raw_events contains all successfully parsed JSON events."""
        ev1 = {
            "type": "response.output_item.done",
            "item": {"type": "message", "content": [{"type": "text", "text": "hi"}]},
        }
        lines = [
            _line(ev1),
            _done(),
        ]
        result = parse_sse_events(lines)
        assert len(result.raw_events) == 1
        assert result.raw_events[0] == ev1

    def test_empty_lines_list(self) -> None:
        """Empty input produces zero-value ParsedResponse."""
        result = parse_sse_events([])
        assert result.content == ""
        assert result.tool_calls == []
        assert result.response_id == ""
        assert result.model == ""
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.raw_events == []

    def test_multiple_text_parts_concatenated(self) -> None:
        """Multiple content items in a single message item are concatenated."""
        lines = [
            _line(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "Hello"},
                            {"type": "output_text", "text": " world"},
                        ],
                    },
                }
            ),
            _done(),
        ]
        result = parse_sse_events(lines)
        assert result.content == "Hello world"

    def test_parsed_response_is_dataclass(self) -> None:
        """ParsedResponse is a dataclass with the expected fields."""
        result = parse_sse_events([])
        # Should be accessible as attributes
        _ = result.content
        _ = result.tool_calls
        _ = result.response_id
        _ = result.model
        _ = result.input_tokens
        _ = result.output_tokens
        _ = result.raw_events


# ---------------------------------------------------------------------------
# TestSSEErrors
# ---------------------------------------------------------------------------


class TestSSEErrors:
    def test_error_event_raises_sse_error(self) -> None:
        """'error' event type raises SSEError."""
        lines = [
            _line(
                {
                    "type": "error",
                    "error": {
                        "message": "Context length exceeded",
                        "code": "context_length_exceeded",
                        "type": "invalid_request_error",
                    },
                }
            ),
        ]
        with pytest.raises(SSEError):
            parse_sse_events(lines)

    def test_response_failed_raises_sse_error(self) -> None:
        """'response.failed' event type raises SSEError."""
        lines = [
            _line(
                {
                    "type": "response.failed",
                    "response": {
                        "error": {
                            "message": "Request failed",
                            "code": "server_error",
                        }
                    },
                }
            ),
        ]
        with pytest.raises(SSEError):
            parse_sse_events(lines)

    def test_response_incomplete_raises_sse_error(self) -> None:
        """'response.incomplete' event type raises SSEError."""
        lines = [
            _line(
                {
                    "type": "response.incomplete",
                    "response": {
                        "error": {
                            "message": "Response was cut short",
                            "code": "max_tokens_exceeded",
                        }
                    },
                }
            ),
        ]
        with pytest.raises(SSEError):
            parse_sse_events(lines)

    def test_sse_error_has_message_attribute(self) -> None:
        """SSEError.message carries the error message from the event."""
        lines = [
            _line(
                {
                    "type": "error",
                    "error": {
                        "message": "Rate limit exceeded",
                        "code": "rate_limit_exceeded",
                        "type": "requests",
                    },
                }
            ),
        ]
        with pytest.raises(SSEError) as exc_info:
            parse_sse_events(lines)
        assert exc_info.value.message == "Rate limit exceeded"

    def test_sse_error_has_code_attribute(self) -> None:
        """SSEError.code carries the error code from the event."""
        lines = [
            _line(
                {
                    "type": "error",
                    "error": {
                        "message": "Rate limit exceeded",
                        "code": "rate_limit_exceeded",
                    },
                }
            ),
        ]
        with pytest.raises(SSEError) as exc_info:
            parse_sse_events(lines)
        assert exc_info.value.code == "rate_limit_exceeded"

    def test_sse_error_has_event_type_attribute(self) -> None:
        """SSEError.event_type is the SSE event type that triggered the error."""
        lines = [
            _line(
                {
                    "type": "error",
                    "error": {
                        "message": "Bad request",
                        "code": "invalid_request",
                    },
                }
            ),
        ]
        with pytest.raises(SSEError) as exc_info:
            parse_sse_events(lines)
        assert exc_info.value.event_type == "error"

    def test_response_failed_event_type_attribute(self) -> None:
        """SSEError.event_type is 'response.failed' for response.failed events."""
        lines = [
            _line(
                {
                    "type": "response.failed",
                    "response": {"error": {"message": "fail", "code": "server_error"}},
                }
            ),
        ]
        with pytest.raises(SSEError) as exc_info:
            parse_sse_events(lines)
        assert exc_info.value.event_type == "response.failed"

    def test_response_incomplete_event_type_attribute(self) -> None:
        """SSEError.event_type is 'response.incomplete' for response.incomplete events."""
        lines = [
            _line(
                {
                    "type": "response.incomplete",
                    "response": {
                        "error": {"message": "incomplete", "code": "max_output_tokens"}
                    },
                }
            ),
        ]
        with pytest.raises(SSEError) as exc_info:
            parse_sse_events(lines)
        assert exc_info.value.event_type == "response.incomplete"

    def test_error_event_no_error_detail(self) -> None:
        """Error event with no error body still raises SSEError with fallback message."""
        lines = [
            _line({"type": "error"}),
        ]
        with pytest.raises(SSEError) as exc_info:
            parse_sse_events(lines)
        # Should have some message even without error detail
        assert exc_info.value.message is not None

    def test_response_failed_no_error_detail(self) -> None:
        """response.failed with no response.error still raises SSEError."""
        lines = [
            _line({"type": "response.failed", "response": {}}),
        ]
        with pytest.raises(SSEError):
            parse_sse_events(lines)

    def test_sse_error_is_exception(self) -> None:
        """SSEError is an Exception subclass."""
        assert issubclass(SSEError, Exception)
