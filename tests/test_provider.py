"""Tests for provider.py — CHATGPT_MODELS catalog, get_info(), and list_models()."""

from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import ANY, AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# TestFallbackCatalog — FALLBACK_MODELS in models.py
# ---------------------------------------------------------------------------


class TestFallbackCatalog:
    """Verify the fallback model entries have the expected shape."""

    def test_fallback_models_nonempty(self) -> None:
        from amplifier_module_provider_openai_chatgpt.models import FALLBACK_MODELS

        assert len(FALLBACK_MODELS) > 0

    def test_fallback_entries_have_slug(self) -> None:
        from amplifier_module_provider_openai_chatgpt.models import FALLBACK_MODELS

        for entry in FALLBACK_MODELS:
            assert "slug" in entry, f"Missing 'slug' in {entry}"

    def test_fallback_contains_gpt_4o(self) -> None:
        from amplifier_module_provider_openai_chatgpt.models import FALLBACK_MODELS

        slugs = [m["slug"] for m in FALLBACK_MODELS]
        assert "gpt-4o" in slugs

    def test_fallback_contains_gpt_52(self) -> None:
        from amplifier_module_provider_openai_chatgpt.models import FALLBACK_MODELS

        slugs = [m["slug"] for m in FALLBACK_MODELS]
        assert "gpt-5.2" in slugs

    def test_fallback_gpt_52_has_fast_tier(self) -> None:
        from amplifier_module_provider_openai_chatgpt.models import FALLBACK_MODELS

        entry = next(m for m in FALLBACK_MODELS if m["slug"] == "gpt-5.2")
        assert "fast" in entry.get("additional_speed_tiers", [])


# ---------------------------------------------------------------------------
# TestGetInfo — provider.get_info()
# ---------------------------------------------------------------------------


class TestGetInfo:
    def _make_provider(self) -> object:
        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        config: dict = {}
        coordinator = MagicMock()
        tokens: dict | None = None
        return ChatGPTProvider(config, coordinator, tokens)

    def test_get_info_id(self) -> None:
        provider = self._make_provider()
        info = provider.get_info()  # type: ignore[union-attr]
        assert info.id == "openai-chatgpt"

    def test_get_info_display_name(self) -> None:
        provider = self._make_provider()
        info = provider.get_info()  # type: ignore[union-attr]
        assert info.display_name == "OpenAI ChatGPT"

    def test_get_info_capabilities_include_streaming(self) -> None:
        provider = self._make_provider()
        info = provider.get_info()  # type: ignore[union-attr]
        assert "streaming" in info.capabilities

    def test_get_info_capabilities_include_tools(self) -> None:
        provider = self._make_provider()
        info = provider.get_info()  # type: ignore[union-attr]
        assert "tools" in info.capabilities


# ---------------------------------------------------------------------------
# TestListModels — provider.list_models() shape validation
# ---------------------------------------------------------------------------


class TestListModels:
    """Verify that list_models() returns well-formed ModelInfo objects.

    Uses FALLBACK_MODELS as the mock return value of fetch_models so the
    tests are deterministic and require no network access.
    """

    def _make_provider(self) -> object:
        from datetime import datetime, timedelta, timezone

        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        config: dict = {}
        coordinator = MagicMock()
        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        tokens = {
            "access_token": "test-token",
            "account_id": "acct-123",
            "expires_at": expires_at,
        }
        return ChatGPTProvider(config, coordinator, tokens)

    def _sample_entries(self) -> list[dict]:
        from amplifier_module_provider_openai_chatgpt.models import FALLBACK_MODELS

        return list(FALLBACK_MODELS)

    @pytest.mark.asyncio
    async def test_list_models_all_have_id(self) -> None:
        provider = self._make_provider()
        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            new=AsyncMock(return_value=self._sample_entries()),
        ):
            models = await provider.list_models()  # type: ignore[union-attr]
        for m in models:
            assert m.id, f"Model missing id: {m}"

    @pytest.mark.asyncio
    async def test_list_models_all_have_display_name(self) -> None:
        provider = self._make_provider()
        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            new=AsyncMock(return_value=self._sample_entries()),
        ):
            models = await provider.list_models()  # type: ignore[union-attr]
        for m in models:
            assert m.display_name, f"Model missing display_name: {m}"

    @pytest.mark.asyncio
    async def test_list_models_all_have_context_window(self) -> None:
        provider = self._make_provider()
        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            new=AsyncMock(return_value=self._sample_entries()),
        ):
            models = await provider.list_models()  # type: ignore[union-attr]
        for m in models:
            assert m.context_window > 0, f"Model missing context_window: {m}"

    @pytest.mark.asyncio
    async def test_list_models_all_have_max_output_tokens(self) -> None:
        provider = self._make_provider()
        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            new=AsyncMock(return_value=self._sample_entries()),
        ):
            models = await provider.list_models()  # type: ignore[union-attr]
        for m in models:
            assert m.max_output_tokens > 0, f"Model missing max_output_tokens: {m}"

    @pytest.mark.asyncio
    async def test_list_models_returns_model_info_objects(self) -> None:
        from amplifier_core import ModelInfo

        provider = self._make_provider()
        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            new=AsyncMock(return_value=self._sample_entries()),
        ):
            models = await provider.list_models()  # type: ignore[union-attr]
        for m in models:
            assert isinstance(m, ModelInfo), f"Expected ModelInfo, got {type(m)}"


# ---------------------------------------------------------------------------
# TestBuildPayload — provider._build_payload()
# ---------------------------------------------------------------------------

REJECTED_PARAMS = {
    "max_output_tokens",
    "temperature",
    "truncation",
    "parallel_tool_calls",
    "include",
}


class TestBuildPayload:
    def _make_provider(self, default_model: str = "gpt-4o") -> object:
        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        config: dict = {"default_model": default_model}
        coordinator = MagicMock()
        tokens: dict | None = None
        return ChatGPTProvider(config, coordinator, tokens)

    def _make_request(self, **kwargs) -> object:  # type: ignore[return]
        from amplifier_core.message_models import ChatRequest, Message

        messages = kwargs.pop("messages", [Message(role="user", content="hello")])
        return ChatRequest(messages=messages, **kwargs)

    # ------------------------------------------------------------------
    # Basic structure
    # ------------------------------------------------------------------

    def test_basic_structure_stream_true(self) -> None:
        """Payload must always have stream=True."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(messages=[Message(role="user", content="hi")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert payload["stream"] is True

    def test_basic_structure_store_false(self) -> None:
        """Payload must always have store=False."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(messages=[Message(role="user", content="hi")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert payload["store"] is False

    def test_basic_structure_uses_input_not_messages(self) -> None:
        """Payload uses 'input' array, not 'messages'."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(messages=[Message(role="user", content="hello")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert "input" in payload
        assert "messages" not in payload

    def test_basic_structure_no_rejected_params(self) -> None:
        """Payload must not contain rejected parameters."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(
            messages=[Message(role="user", content="hi")],
            max_output_tokens=1000,
            temperature=0.7,
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        for param in REJECTED_PARAMS:
            assert param not in payload, f"Rejected param '{param}' found in payload"

    def test_basic_structure_has_model(self) -> None:
        """Payload must include a model field."""
        from amplifier_core.message_models import Message

        provider = self._make_provider(default_model="gpt-4o")
        request = self._make_request(messages=[Message(role="user", content="hi")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert "model" in payload
        assert payload["model"] == "gpt-4o"

    # ------------------------------------------------------------------
    # System message → instructions
    # ------------------------------------------------------------------

    def test_system_message_becomes_instructions(self) -> None:
        """System message content extracted as top-level 'instructions'."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hello"),
            ]
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert payload.get("instructions") == "You are a helpful assistant."

    def test_system_message_not_in_input(self) -> None:
        """System message must not appear in the input array."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(
            messages=[
                Message(role="system", content="System prompt here."),
                Message(role="user", content="Hello"),
            ]
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        for item in payload["input"]:
            role = item.get("role")
            assert role not in ("system", "developer"), (
                f"System/developer message found in input array: {item}"
            )

    # ------------------------------------------------------------------
    # -fast suffix → priority service_tier
    # ------------------------------------------------------------------

    def test_fast_suffix_stripped_from_model(self) -> None:
        """Model name with -fast suffix has suffix stripped in payload."""
        from amplifier_core.message_models import Message

        provider = self._make_provider(default_model="gpt-5.4-fast")
        request = self._make_request(messages=[Message(role="user", content="hi")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert payload["model"] == "gpt-5.4"

    def test_fast_suffix_adds_priority_service_tier(self) -> None:
        """-fast suffix model adds service_tier='priority'."""
        from amplifier_core.message_models import Message

        provider = self._make_provider(default_model="gpt-5.4-fast")
        request = self._make_request(messages=[Message(role="user", content="hi")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert payload.get("service_tier") == "priority"

    # ------------------------------------------------------------------
    # Non-fast model — no service_tier
    # ------------------------------------------------------------------

    def test_non_fast_model_no_service_tier(self) -> None:
        """Non -fast model must not include service_tier key."""
        from amplifier_core.message_models import Message

        provider = self._make_provider(default_model="gpt-4o")
        request = self._make_request(messages=[Message(role="user", content="hi")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert "service_tier" not in payload

    # ------------------------------------------------------------------
    # Model override
    # ------------------------------------------------------------------

    def test_request_model_overrides_default(self) -> None:
        """Request model field overrides provider default_model."""
        from amplifier_core.message_models import Message

        provider = self._make_provider(default_model="gpt-4o")
        request = self._make_request(
            messages=[Message(role="user", content="hi")],
            model="gpt-5.4",
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert payload["model"] == "gpt-5.4"

    def test_request_model_fast_overrides_with_service_tier(self) -> None:
        """Request model with -fast suffix overrides default and adds service_tier."""
        from amplifier_core.message_models import Message

        provider = self._make_provider(default_model="gpt-4o")
        request = self._make_request(
            messages=[Message(role="user", content="hi")],
            model="gpt-5.4-fast",
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert payload["model"] == "gpt-5.4"
        assert payload.get("service_tier") == "priority"

    # ------------------------------------------------------------------
    # Tools conversion
    # ------------------------------------------------------------------

    def test_tools_converted_to_responses_api_format(self) -> None:
        """Tools converted to {type, name, description, parameters} format."""
        from amplifier_core.message_models import Message, ToolSpec

        provider = self._make_provider()
        request = self._make_request(
            messages=[Message(role="user", content="call a tool")],
            tools=[
                ToolSpec(
                    name="get_weather",
                    description="Gets the weather",
                    parameters={"type": "object", "properties": {}},
                )
            ],
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert "tools" in payload
        tools = payload["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["name"] == "get_weather"
        assert tools[0]["description"] == "Gets the weather"
        assert tools[0]["parameters"] == {"type": "object", "properties": {}}

    def test_tools_adds_tool_choice_auto(self) -> None:
        """When tools present, tool_choice should be 'auto'."""
        from amplifier_core.message_models import Message, ToolSpec

        provider = self._make_provider()
        request = self._make_request(
            messages=[Message(role="user", content="hi")],
            tools=[ToolSpec(name="foo", parameters={"type": "object"})],
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert payload.get("tool_choice") == "auto"

    def test_no_tools_no_tool_choice(self) -> None:
        """Without tools, tool_choice must not appear in payload."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(messages=[Message(role="user", content="hi")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert "tool_choice" not in payload

    # ------------------------------------------------------------------
    # Structured content (user message with TextBlock)
    # ------------------------------------------------------------------

    def test_user_message_string_content_converted(self) -> None:
        """String user content converted to input_text format."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(messages=[Message(role="user", content="hello")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        user_items = [i for i in payload["input"] if i.get("role") == "user"]
        assert len(user_items) == 1
        content = user_items[0]["content"]
        assert content == [{"type": "input_text", "text": "hello"}]

    def test_user_message_text_block_content_converted(self) -> None:
        """TextBlock user content converted to input_text format."""
        from amplifier_core.message_models import Message, TextBlock

        provider = self._make_provider()
        request = self._make_request(
            messages=[Message(role="user", content=[TextBlock(text="world")])]
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        user_items = [i for i in payload["input"] if i.get("role") == "user"]
        assert len(user_items) == 1
        content = user_items[0]["content"]
        assert {"type": "input_text", "text": "world"} in content

    # ------------------------------------------------------------------
    # Tool result conversion
    # ------------------------------------------------------------------

    def test_tool_result_converted_to_function_call_output(self) -> None:
        """ToolResultBlock → {type: function_call_output, call_id, output}."""
        from amplifier_core.message_models import Message, ToolResultBlock

        provider = self._make_provider()
        request = self._make_request(
            messages=[
                Message(role="user", content="run tool"),
                Message(
                    role="tool",
                    content=[
                        ToolResultBlock(
                            tool_call_id="call_123",
                            output="the result",
                        )
                    ],
                ),
            ]
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        tool_outputs = [
            i for i in payload["input"] if i.get("type") == "function_call_output"
        ]
        assert len(tool_outputs) == 1
        assert tool_outputs[0]["call_id"] == "call_123"
        assert tool_outputs[0]["output"] == "the result"

    def test_assistant_tool_call_converted_to_function_call(self) -> None:
        """ToolCallBlock in assistant message → {type: function_call, call_id, name, arguments}."""
        from amplifier_core.message_models import Message, ToolCallBlock

        provider = self._make_provider()
        request = self._make_request(
            messages=[
                Message(
                    role="assistant",
                    content=[
                        ToolCallBlock(
                            id="call_abc",
                            name="search",
                            input={"query": "test"},
                        )
                    ],
                )
            ]
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        func_calls = [i for i in payload["input"] if i.get("type") == "function_call"]
        assert len(func_calls) == 1
        assert func_calls[0]["call_id"] == "call_abc"
        assert func_calls[0]["name"] == "search"
        # arguments must be a JSON string of the input dict
        args = func_calls[0]["arguments"]
        assert json.loads(args) == {"query": "test"}

    # ------------------------------------------------------------------
    # Reasoning effort
    # ------------------------------------------------------------------

    def test_reasoning_effort_adds_reasoning_block(self) -> None:
        """reasoning_effort in request → {reasoning: {effort, summary: 'detailed'}}."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(
            messages=[Message(role="user", content="think")],
            reasoning_effort="high",
        )
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert "reasoning" in payload
        assert payload["reasoning"]["effort"] == "high"
        assert payload["reasoning"]["summary"] == "detailed"

    def test_no_reasoning_effort_no_reasoning_block(self) -> None:
        """Without reasoning_effort, no reasoning key in payload."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(messages=[Message(role="user", content="hi")])
        payload = provider._build_payload(request)  # type: ignore[union-attr]
        assert "reasoning" not in payload


# ---------------------------------------------------------------------------
# TestBuildHeaders — provider._build_headers()
# ---------------------------------------------------------------------------


class TestBuildHeaders:
    def _make_provider(
        self, tokens: dict | None = None, config: dict | None = None
    ) -> object:
        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        if config is None:
            config = {}
        coordinator = MagicMock()
        return ChatGPTProvider(config, coordinator, tokens)

    def _valid_tokens(self) -> dict:
        return {"access_token": "test-access-tok", "account_id": "acct-123"}

    # ------------------------------------------------------------------
    # Happy path: all 6 required headers present with correct values
    # ------------------------------------------------------------------

    def test_authorization_header(self) -> None:
        """Authorization header must be 'Bearer {access_token}'."""
        provider = self._make_provider(tokens=self._valid_tokens())
        headers = provider._build_headers()  # type: ignore[union-attr]
        assert headers["Authorization"] == "Bearer test-access-tok"

    def test_chatgpt_account_id_header(self) -> None:
        """ChatGPT-Account-Id header must equal account_id from tokens."""
        provider = self._make_provider(tokens=self._valid_tokens())
        headers = provider._build_headers()  # type: ignore[union-attr]
        assert headers["ChatGPT-Account-Id"] == "acct-123"

    def test_openai_beta_header(self) -> None:
        """OpenAI-Beta header must be 'responses=v1'."""
        provider = self._make_provider(tokens=self._valid_tokens())
        headers = provider._build_headers()  # type: ignore[union-attr]
        assert headers["OpenAI-Beta"] == "responses=v1"

    def test_openai_originator_header(self) -> None:
        """OpenAI-Originator header must be 'codex'."""
        provider = self._make_provider(tokens=self._valid_tokens())
        headers = provider._build_headers()  # type: ignore[union-attr]
        assert headers["OpenAI-Originator"] == "codex"

    def test_content_type_header(self) -> None:
        """Content-Type header must be 'application/json'."""
        provider = self._make_provider(tokens=self._valid_tokens())
        headers = provider._build_headers()  # type: ignore[union-attr]
        assert headers["Content-Type"] == "application/json"

    def test_accept_header(self) -> None:
        """accept header must be 'text/event-stream'."""
        provider = self._make_provider(tokens=self._valid_tokens())
        headers = provider._build_headers()  # type: ignore[union-attr]
        assert headers["accept"] == "text/event-stream"

    def test_all_six_fields_present(self) -> None:
        """Headers dict must have exactly the 6 required fields."""
        provider = self._make_provider(tokens=self._valid_tokens())
        headers = provider._build_headers()  # type: ignore[union-attr]
        expected_keys = {
            "Authorization",
            "ChatGPT-Account-Id",
            "OpenAI-Beta",
            "OpenAI-Originator",
            "Content-Type",
            "accept",
        }
        assert set(headers.keys()) == expected_keys

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    def test_missing_access_token_raises_value_error(self) -> None:
        """Missing access_token raises ValueError with 'No valid OAuth tokens'."""
        tokens = {"account_id": "acct-123"}  # no access_token
        provider = self._make_provider(tokens=tokens)
        with pytest.raises(ValueError, match="No valid OAuth tokens"):
            provider._build_headers()  # type: ignore[union-attr]

    def test_none_tokens_raises_value_error(self) -> None:
        """tokens=None raises ValueError with 'No valid OAuth tokens'."""
        provider = self._make_provider(tokens=None)
        with pytest.raises(ValueError, match="No valid OAuth tokens"):
            provider._build_headers()  # type: ignore[union-attr]

    def test_missing_account_id_raises_value_error(self) -> None:
        """Missing account_id raises ValueError mentioning 'account_id'."""
        tokens = {"access_token": "test-access-tok"}  # no account_id
        provider = self._make_provider(tokens=tokens)
        with pytest.raises(ValueError, match="account_id"):
            provider._build_headers()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# TestComplete — provider.complete()
# ---------------------------------------------------------------------------


def _make_sse_lines(
    text: str | None = None,
    tool_calls: list[dict] | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> list[str]:
    """Build minimal SSE line list simulating a ChatGPT streaming response."""
    events = []

    # response.created
    events.append(
        json.dumps(
            {
                "type": "response.created",
                "response": {"id": "resp_test", "model": "gpt-4o"},
            }
        )
    )

    # Text content
    if text is not None:
        events.append(
            json.dumps(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "content": [{"type": "output_text", "text": text}],
                    },
                }
            )
        )

    # Tool calls
    for tc in tool_calls or []:
        events.append(
            json.dumps(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": tc["id"],
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
            )
        )

    # response.done with usage
    events.append(
        json.dumps(
            {
                "type": "response.done",
                "response": {
                    "id": "resp_test",
                    "model": "gpt-4o",
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                },
            }
        )
    )

    lines = [f"data: {e}" for e in events]
    lines.append("data: [DONE]")
    return lines


class _MockStreamResponse:
    """Async mock for an httpx streaming response."""

    def __init__(self, lines: list[str], status_code: int = 200) -> None:
        self.status_code = status_code
        self._lines = lines

    async def aiter_lines(self):  # type: ignore[return]
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b"HTTP error body"


class _AsyncCM:
    """Simple async context manager wrapping a value."""

    def __init__(self, value: object) -> None:
        self._value = value

    async def __aenter__(self) -> object:
        return self._value

    async def __aexit__(self, *args: object) -> None:
        pass


def _make_sse_response(lines: list[str], status_code: int = 200) -> "_AsyncCM":
    """Return an async-context-manager mock for httpx.AsyncClient.

    Usage::

        with patch("...httpx.AsyncClient") as MockClient:
            MockClient.return_value = _make_sse_response(lines)
            result = await provider.complete(request)
    """
    response = _MockStreamResponse(lines, status_code)
    mock_client = MagicMock()
    mock_client.stream.return_value = _AsyncCM(response)
    return _AsyncCM(mock_client)


class TestComplete:
    """Tests for ChatGPTProvider.complete()."""

    def _make_provider(self, raw: bool = False) -> object:
        from datetime import datetime, timedelta, timezone

        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        config: dict = {"raw": raw, "default_model": "gpt-4o"}
        coordinator = MagicMock()
        coordinator.hooks.emit = AsyncMock()
        # Tokens that pass is_token_valid()
        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        tokens = {
            "access_token": "test-access-token",
            "account_id": "acct-123",
            "expires_at": expires_at,
        }
        return ChatGPTProvider(config, coordinator, tokens)

    def _make_request(self, **kwargs: object) -> object:  # type: ignore[return]
        from amplifier_core.message_models import ChatRequest, Message

        messages = kwargs.pop("messages", [Message(role="user", content="hello")])
        return ChatRequest(messages=messages, **kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Simple text completion
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_simple_text_completion_content(self) -> None:
        """complete() returns ChatResponse with correct text content."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(
            text="Hello, world!", input_tokens=10, output_tokens=5
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            result = await provider.complete(request)  # type: ignore[union-attr]

        from amplifier_core.message_models import TextBlock

        assert len(result.content) == 1
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "Hello, world!"

    @pytest.mark.asyncio
    async def test_simple_text_completion_usage(self) -> None:
        """complete() returns correct usage counts."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(text="Hello!", input_tokens=10, output_tokens=5)

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            result = await provider.complete(request)  # type: ignore[union-attr]

        assert result.usage is not None
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.usage.total_tokens == 15

    @pytest.mark.asyncio
    async def test_simple_text_completion_finish_reason_stop(self) -> None:
        """Text-only response has finish_reason='stop'."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(text="Done.")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            result = await provider.complete(request)  # type: ignore[union-attr]

        assert result.finish_reason == "stop"

    # ------------------------------------------------------------------
    # Tool call response
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_tool_call_response_name(self) -> None:
        """Tool call response populates tool name correctly."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(
            tool_calls=[
                {
                    "id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"city": "London"}',
                }
            ]
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            result = await provider.complete(request)  # type: ignore[union-attr]

        from amplifier_core.message_models import ToolCallBlock

        tool_blocks = [b for b in result.content if isinstance(b, ToolCallBlock)]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].name == "get_weather"

    @pytest.mark.asyncio
    async def test_tool_call_response_id(self) -> None:
        """Tool call response populates call ID correctly."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(
            tool_calls=[
                {
                    "id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"city": "London"}',
                }
            ]
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            result = await provider.complete(request)  # type: ignore[union-attr]

        from amplifier_core.message_models import ToolCallBlock

        tool_blocks = [b for b in result.content if isinstance(b, ToolCallBlock)]
        assert tool_blocks[0].id == "call_abc"

    @pytest.mark.asyncio
    async def test_tool_call_response_arguments_dict(self) -> None:
        """Tool call arguments are parsed to a dict."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(
            tool_calls=[
                {
                    "id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"city": "London"}',
                }
            ]
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            result = await provider.complete(request)  # type: ignore[union-attr]

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].arguments == {"city": "London"}

    @pytest.mark.asyncio
    async def test_tool_call_response_finish_reason_tool_calls(self) -> None:
        """Response with tool calls has finish_reason='tool_calls'."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(
            tool_calls=[
                {
                    "id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"city": "London"}',
                }
            ]
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            result = await provider.complete(request)  # type: ignore[union-attr]

        assert result.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_tool_call_arguments_fallback_on_bad_json(self) -> None:
        """Invalid JSON arguments fall back to {'_raw': ...}."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(
            tool_calls=[
                {"id": "call_abc", "name": "foo", "arguments": "not-valid-json"}
            ]
        )

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            result = await provider.complete(request)  # type: ignore[union-attr]

        assert result.tool_calls is not None
        assert "_raw" in result.tool_calls[0].arguments
        assert result.tool_calls[0].arguments["_raw"] == "not-valid-json"

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_emits_llm_request_event_provider(self) -> None:
        """complete() emits llm:request with provider='openai-chatgpt'."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            await provider.complete(request)  # type: ignore[union-attr]

        coordinator = provider._coordinator  # type: ignore[union-attr]
        calls = coordinator.hooks.emit.call_args_list
        request_calls = [c for c in calls if c.args[0] == "llm:request"]
        assert len(request_calls) >= 1
        event_data = request_calls[0].args[1]
        assert event_data["provider"] == "openai-chatgpt"

    @pytest.mark.asyncio
    async def test_emits_llm_request_event_message_count(self) -> None:
        """complete() emits llm:request with correct message_count."""
        from amplifier_core.message_models import Message

        provider = self._make_provider()
        request = self._make_request(
            messages=[
                Message(role="user", content="hello"),
                Message(role="user", content="world"),
            ]
        )
        sse_lines = _make_sse_lines(text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            await provider.complete(request)  # type: ignore[union-attr]

        coordinator = provider._coordinator  # type: ignore[union-attr]
        calls = coordinator.hooks.emit.call_args_list
        request_calls = [c for c in calls if c.args[0] == "llm:request"]
        event_data = request_calls[0].args[1]
        assert event_data["message_count"] == 2

    @pytest.mark.asyncio
    async def test_emits_llm_response_event_status_ok(self) -> None:
        """complete() emits llm:response with status='ok' on success."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            await provider.complete(request)  # type: ignore[union-attr]

        coordinator = provider._coordinator  # type: ignore[union-attr]
        calls = coordinator.hooks.emit.call_args_list
        response_calls = [c for c in calls if c.args[0] == "llm:response"]
        assert len(response_calls) >= 1
        event_data = response_calls[0].args[1]
        assert event_data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_emits_llm_response_event_duration_ms(self) -> None:
        """complete() emits llm:response with duration_ms present."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            await provider.complete(request)  # type: ignore[union-attr]

        coordinator = provider._coordinator  # type: ignore[union-attr]
        calls = coordinator.hooks.emit.call_args_list
        response_calls = [c for c in calls if c.args[0] == "llm:response"]
        event_data = response_calls[0].args[1]
        assert "duration_ms" in event_data
        assert isinstance(event_data["duration_ms"], float)

    @pytest.mark.asyncio
    async def test_emits_llm_response_event_provider(self) -> None:
        """complete() emits llm:response with provider='openai-chatgpt'."""
        provider = self._make_provider()
        request = self._make_request()
        sse_lines = _make_sse_lines(text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            await provider.complete(request)  # type: ignore[union-attr]

        coordinator = provider._coordinator  # type: ignore[union-attr]
        calls = coordinator.hooks.emit.call_args_list
        response_calls = [c for c in calls if c.args[0] == "llm:response"]
        event_data = response_calls[0].args[1]
        assert event_data["provider"] == "openai-chatgpt"

    # ------------------------------------------------------------------
    # Raw mode
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_raw_mode_request_event_includes_payload(self) -> None:
        """Raw mode: llm:request event includes raw payload."""
        provider = self._make_provider(raw=True)
        request = self._make_request()
        sse_lines = _make_sse_lines(text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            await provider.complete(request)  # type: ignore[union-attr]

        coordinator = provider._coordinator  # type: ignore[union-attr]
        calls = coordinator.hooks.emit.call_args_list
        request_calls = [c for c in calls if c.args[0] == "llm:request"]
        event_data = request_calls[0].args[1]
        assert "raw" in event_data

    @pytest.mark.asyncio
    async def test_raw_mode_response_event_includes_raw_events(self) -> None:
        """Raw mode: llm:response event includes raw_events from SSE."""
        provider = self._make_provider(raw=True)
        request = self._make_request()
        sse_lines = _make_sse_lines(text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            await provider.complete(request)  # type: ignore[union-attr]

        coordinator = provider._coordinator  # type: ignore[union-attr]
        calls = coordinator.hooks.emit.call_args_list
        response_calls = [c for c in calls if c.args[0] == "llm:response"]
        event_data = response_calls[0].args[1]
        assert "raw" in event_data

    @pytest.mark.asyncio
    async def test_non_raw_mode_request_no_payload(self) -> None:
        """Non-raw mode: llm:request event does NOT include payload."""
        provider = self._make_provider(raw=False)
        request = self._make_request()
        sse_lines = _make_sse_lines(text="hi")

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(sse_lines)
            await provider.complete(request)  # type: ignore[union-attr]

        coordinator = provider._coordinator  # type: ignore[union-attr]
        calls = coordinator.hooks.emit.call_args_list
        request_calls = [c for c in calls if c.args[0] == "llm:request"]
        event_data = request_calls[0].args[1]
        assert "payload" not in event_data


# ---------------------------------------------------------------------------
# TestCompleteErrors — error handling paths in complete()
# ---------------------------------------------------------------------------


class TestCompleteErrors:
    """Verify error handling in complete(): error events emitted before exceptions propagate."""

    def _make_provider_with_tokens(self) -> object:
        """Create ChatGPTProvider with valid OAuth tokens and a mock coordinator."""
        from datetime import datetime, timedelta, timezone

        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        tokens = {
            "access_token": "test-access-token",
            "account_id": "acct-123",
            "expires_at": expires_at,
        }
        coordinator = MagicMock()
        coordinator.hooks.emit = AsyncMock()
        return ChatGPTProvider({"default_model": "gpt-4o"}, coordinator, tokens)

    def _make_request(self) -> object:  # type: ignore[return]
        from amplifier_core.message_models import ChatRequest, Message

        return ChatRequest(messages=[Message(role="user", content="hello")])

    @pytest.mark.asyncio
    async def test_sse_error_event_emits_error_status_then_raises(self) -> None:
        """SSE error event inside stream emits llm:response with status='error', then raises SSEError."""
        from amplifier_module_provider_openai_chatgpt._sse import SSEError

        provider = self._make_provider_with_tokens()
        request = self._make_request()

        # SSE lines containing an error event
        error_lines = [
            "data: "
            + json.dumps(
                {
                    "type": "error",
                    "error": {
                        "message": "Something went wrong",
                        "code": "server_error",
                    },
                }
            ),
            "data: [DONE]",
        ]

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.httpx.AsyncClient"
        ) as MockClient:
            MockClient.return_value = _make_sse_response(error_lines)
            with pytest.raises(SSEError):
                await provider.complete(request)  # type: ignore[union-attr]

        coordinator = provider._coordinator  # type: ignore[union-attr]
        calls = coordinator.hooks.emit.call_args_list
        error_response_calls = [
            c
            for c in calls
            if c.args[0] == "llm:response" and c.args[1].get("status") == "error"
        ]
        assert len(error_response_calls) >= 1, (
            "Expected at least one llm:response event with status='error'"
        )

    @pytest.mark.asyncio
    async def test_missing_tokens_raises_value_error(self) -> None:
        """Provider with no valid tokens raises ValueError before making any HTTP request."""
        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        coordinator = MagicMock()
        coordinator.hooks.emit = AsyncMock()
        provider = ChatGPTProvider(
            {"default_model": "gpt-4o"}, coordinator, tokens=None
        )
        request = self._make_request()

        with patch(
            "amplifier_module_provider_openai_chatgpt.oauth.load_tokens",
            return_value=None,
        ):
            with pytest.raises(ValueError):
                await provider.complete(request)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# TestMount — module-level mount() function
# ---------------------------------------------------------------------------


class TestMount:
    """Tests for the module-level mount() function in __init__.py."""

    @pytest.mark.asyncio
    async def test_mount_with_valid_token_file_succeeds(self) -> None:
        """mount() with valid tokens calls coordinator.mount once with 'providers'
        and name='openai-chatgpt', and returns a callable cleanup."""
        from datetime import datetime, timedelta, timezone

        from amplifier_module_provider_openai_chatgpt import mount

        coordinator = MagicMock()
        # coordinator.mount is awaited in __init__.py, so it must be an AsyncMock
        coordinator.mount = AsyncMock()
        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        valid_tokens = {
            "access_token": "test-access-token",
            "account_id": "acct-123",
            "expires_at": expires_at,
        }

        config = {"token_file_path": "/tmp/fake_tokens.json"}

        with patch(
            "amplifier_module_provider_openai_chatgpt.load_tokens",
            return_value=valid_tokens,
        ):
            with patch(
                "amplifier_module_provider_openai_chatgpt.is_token_valid",
                return_value=True,
            ):
                cleanup = await mount(coordinator, config)

        coordinator.mount.assert_called_once_with(
            "providers", ANY, name="openai-chatgpt"
        )
        assert callable(cleanup)

    @pytest.mark.asyncio
    async def test_mount_returns_none_no_tokens_login_disabled(self) -> None:
        """mount() returns None when no valid tokens and login_on_mount=False.
        coordinator.mount must not be called."""
        from amplifier_module_provider_openai_chatgpt import mount

        coordinator = MagicMock()
        config = {"token_file_path": "/tmp/fake_tokens.json", "login_on_mount": False}

        with patch(
            "amplifier_module_provider_openai_chatgpt.load_tokens",
            return_value=None,
        ):
            with patch(
                "amplifier_module_provider_openai_chatgpt.is_token_valid",
                return_value=False,
            ):
                result = await mount(coordinator, config)

        assert result is None
        coordinator.mount.assert_not_called()


# ---------------------------------------------------------------------------
# TestListModelsDynamic — caching logic in _get_catalog() / list_models()
# ---------------------------------------------------------------------------


class TestListModelsDynamic:
    """Tests for the cached model fetching introduced by _get_catalog().

    Follows the same token-mocking approach as TestComplete: valid tokens
    are set on the provider so _ensure_valid_tokens() passes immediately,
    and fetch_models is patched to control catalog fetch behaviour.
    """

    def _make_provider(self, models_cache_ttl: float = 3600.0) -> object:
        from datetime import datetime, timedelta, timezone

        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        expires_at = (datetime.now(tz=timezone.utc) + timedelta(hours=1)).isoformat()
        tokens = {
            "access_token": "test-access-token",
            "account_id": "acct-123",
            "expires_at": expires_at,
        }
        config = {"models_cache_ttl": models_cache_ttl, "default_model": "gpt-4o"}
        coordinator = MagicMock()
        return ChatGPTProvider(config, coordinator, tokens)

    def _sample_entries(self) -> list[dict]:
        """Single API-visible model entry (no fast tier) for deterministic tests."""
        return [
            {
                "slug": "gpt-4o",
                "display_name": "GPT-4o",
                "context_window": 128000,
                "supported_in_api": True,
                "visibility": "visible",
                "additional_speed_tiers": [],
                "supported_reasoning_levels": [],
                "default_reasoning_level": None,
            }
        ]

    # ------------------------------------------------------------------
    # First-call fetch
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_models_fetches_on_first_call(self) -> None:
        """First call to list_models() fetches from API exactly once."""
        provider = self._make_provider()
        mock_fetch = AsyncMock(return_value=self._sample_entries())

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            mock_fetch,
        ):
            result = await provider.list_models()  # type: ignore[union-attr]

        mock_fetch.assert_awaited_once()
        assert len(result) == 1
        assert result[0].id == "gpt-4o"

    # ------------------------------------------------------------------
    # Cache hit on second call
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_models_uses_cache_on_second_call(self) -> None:
        """Second call within TTL returns cached result; fetch_models called once."""
        provider = self._make_provider()
        mock_fetch = AsyncMock(return_value=self._sample_entries())

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            mock_fetch,
        ):
            await provider.list_models()  # type: ignore[union-attr]
            await provider.list_models()  # type: ignore[union-attr]

        mock_fetch.assert_awaited_once()

    # ------------------------------------------------------------------
    # Fallback on failure, cache stays empty
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_models_falls_back_on_failure(self) -> None:
        """When fetch_models raises, fallback models are returned and cache stays None."""
        from amplifier_module_provider_openai_chatgpt.models import FALLBACK_MODELS

        provider = self._make_provider()
        mock_fetch = AsyncMock(side_effect=ValueError("simulated API error"))

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            mock_fetch,
        ):
            result = await provider.list_models()  # type: ignore[union-attr]

        # All fallback slugs must appear in the result (some produce -fast variants).
        fallback_slugs = {entry["slug"] for entry in FALLBACK_MODELS}
        result_ids = {m.id for m in result}
        for slug in fallback_slugs:
            assert slug in result_ids or f"{slug}-fast" in result_ids, (
                f"Expected fallback slug {slug!r} (or its -fast variant) in {result_ids}"
            )

        # Cache must NOT be populated on failure — next call should retry.
        assert provider._models_cache is None  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # TTL expiry causes re-fetch
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_models_ttl_expiry_refetches(self) -> None:
        """With models_cache_ttl=0, each call triggers a fresh fetch."""
        provider = self._make_provider(models_cache_ttl=0.0)
        mock_fetch = AsyncMock(return_value=self._sample_entries())

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            mock_fetch,
        ):
            await provider.list_models()  # type: ignore[union-attr]
            await provider.list_models()  # type: ignore[union-attr]

        assert mock_fetch.await_count == 2, (
            f"Expected 2 fetches with TTL=0, got {mock_fetch.await_count}"
        )

    # ------------------------------------------------------------------
    # Empty catalog triggers fallback
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_models_falls_back_on_empty_catalog(self) -> None:
        """When fetch_models returns [], fallback models are returned and cache stays None."""
        provider = self._make_provider()
        mock_fetch = AsyncMock(return_value=[])

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            mock_fetch,
        ):
            result = await provider.list_models()  # type: ignore[union-attr]

        assert len(result) > 0, "Expected non-empty fallback models"
        # Cache must NOT be populated — next call should retry the live fetch.
        assert provider._models_cache is None  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Concurrent calls fetch exactly once
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_concurrent_list_models_fetches_once(self) -> None:
        """5 concurrent list_models() calls result in exactly one fetch_models invocation."""
        provider = self._make_provider()
        mock_fetch = AsyncMock(return_value=self._sample_entries())

        with patch(
            "amplifier_module_provider_openai_chatgpt.provider.fetch_models",
            mock_fetch,
        ):
            results = await asyncio.gather(
                provider.list_models(),  # type: ignore[union-attr]
                provider.list_models(),  # type: ignore[union-attr]
                provider.list_models(),  # type: ignore[union-attr]
                provider.list_models(),  # type: ignore[union-attr]
                provider.list_models(),  # type: ignore[union-attr]
            )

        mock_fetch.assert_awaited_once()
        # All 5 callers should receive the same non-empty result.
        for result in results:
            assert len(result) > 0
