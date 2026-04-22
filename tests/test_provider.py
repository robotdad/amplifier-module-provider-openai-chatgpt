"""Tests for provider.py — CHATGPT_MODELS catalog, get_info(), and list_models()."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# TestModelCatalog — raw CHATGPT_MODELS dict list
# ---------------------------------------------------------------------------


class TestModelCatalog:
    def test_catalog_is_nonempty(self) -> None:
        from amplifier_module_provider_openai_chatgpt.provider import CHATGPT_MODELS

        assert len(CHATGPT_MODELS) > 0

    def test_catalog_entries_have_required_fields(self) -> None:
        from amplifier_module_provider_openai_chatgpt.provider import CHATGPT_MODELS

        for entry in CHATGPT_MODELS:
            assert "name" in entry, f"Missing 'name' in {entry}"
            assert "context_window" in entry, f"Missing 'context_window' in {entry}"
            assert "max_output_tokens" in entry, (
                f"Missing 'max_output_tokens' in {entry}"
            )

    def test_catalog_contains_known_model_gpt54(self) -> None:
        from amplifier_module_provider_openai_chatgpt.provider import CHATGPT_MODELS

        names = [m["name"] for m in CHATGPT_MODELS]
        assert "gpt-5.4" in names

    def test_catalog_contains_known_model_o4_mini(self) -> None:
        from amplifier_module_provider_openai_chatgpt.provider import CHATGPT_MODELS

        names = [m["name"] for m in CHATGPT_MODELS]
        assert "o4-mini" in names

    def test_catalog_contains_known_model_gpt4o(self) -> None:
        from amplifier_module_provider_openai_chatgpt.provider import CHATGPT_MODELS

        names = [m["name"] for m in CHATGPT_MODELS]
        assert "gpt-4o" in names

    def test_gpt54_context_window(self) -> None:
        from amplifier_module_provider_openai_chatgpt.provider import CHATGPT_MODELS

        entry = next(m for m in CHATGPT_MODELS if m["name"] == "gpt-5.4")
        assert entry["context_window"] == 272000

    def test_gpt54_max_output_tokens(self) -> None:
        from amplifier_module_provider_openai_chatgpt.provider import CHATGPT_MODELS

        entry = next(m for m in CHATGPT_MODELS if m["name"] == "gpt-5.4")
        assert entry["max_output_tokens"] == 128000

    def test_gpt4o_context_window_and_max_output(self) -> None:
        from amplifier_module_provider_openai_chatgpt.provider import CHATGPT_MODELS

        entry = next(m for m in CHATGPT_MODELS if m["name"] == "gpt-4o")
        assert entry["context_window"] == 128000
        assert entry["max_output_tokens"] == 16384

    def test_o4_mini_context_window(self) -> None:
        from amplifier_module_provider_openai_chatgpt.provider import CHATGPT_MODELS

        entry = next(m for m in CHATGPT_MODELS if m["name"] == "o4-mini")
        assert entry["context_window"] == 200000


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
# TestListModels — provider.list_models()
# ---------------------------------------------------------------------------


class TestListModels:
    def _make_provider(self) -> object:
        from amplifier_module_provider_openai_chatgpt.provider import ChatGPTProvider

        config: dict = {}
        coordinator = MagicMock()
        tokens: dict | None = None
        return ChatGPTProvider(config, coordinator, tokens)

    def test_list_models_correct_count(self) -> None:
        from amplifier_module_provider_openai_chatgpt.provider import CHATGPT_MODELS

        provider = self._make_provider()
        models = provider.list_models()  # type: ignore[union-attr]
        assert len(models) == len(CHATGPT_MODELS)

    def test_list_models_all_have_id(self) -> None:
        provider = self._make_provider()
        models = provider.list_models()  # type: ignore[union-attr]
        for m in models:
            assert m.id, f"Model missing id: {m}"

    def test_list_models_all_have_display_name(self) -> None:
        provider = self._make_provider()
        models = provider.list_models()  # type: ignore[union-attr]
        for m in models:
            assert m.display_name, f"Model missing display_name: {m}"

    def test_list_models_all_have_context_window(self) -> None:
        provider = self._make_provider()
        models = provider.list_models()  # type: ignore[union-attr]
        for m in models:
            assert m.context_window > 0, f"Model missing context_window: {m}"

    def test_list_models_all_have_max_output_tokens(self) -> None:
        provider = self._make_provider()
        models = provider.list_models()  # type: ignore[union-attr]
        for m in models:
            assert m.max_output_tokens > 0, f"Model missing max_output_tokens: {m}"

    def test_list_models_returns_model_info_objects(self) -> None:
        from amplifier_core import ModelInfo

        provider = self._make_provider()
        models = provider.list_models()  # type: ignore[union-attr]
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
