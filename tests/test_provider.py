"""Tests for provider.py — CHATGPT_MODELS catalog, get_info(), and list_models()."""

from __future__ import annotations

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
